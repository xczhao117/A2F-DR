from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
import os
import traceback
import pdb
import warnings
import dill
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from collections import defaultdict, namedtuple
import torch
warnings.filterwarnings('ignore')

Args = namedtuple("Args", ['datadir'])
# cert_words = ['0054-3194', '0517-1010', '66213-423', '66213-425', '0225-0295', '66213-423', '66213-425']
cert_words = ['0456-4020', '54838-540', '60505-2519', '0456-2010', '51079-543']

def _get_metric(datadir, modelname):
    args = Args(datadir=datadir)
    if args.MIMIC == 4:
        data_path = os.path.join(args.datadir, 'records_final_4.pkl')
        voc_path = os.path.join(args.datadir, 'voc_final_4.pkl')
    else:
        data_path = os.path.join(args.datadir, 'records_final.pkl')
        voc_path = os.path.join(args.datadir, 'voc_final.pkl')        
    voc = dill.load(open(voc_path, 'rb'))
    med_voc = voc['med_voc']
    data = dill.load(open(data_path, 'rb'))

    main_met_obj = Metrics(data, med_voc, args)
    loadpath = os.path.join("saved", modelname, 'test_gt_pred_prob.pkl')
    gt, pred, prob = dill.load(open(loadpath, 'rb'))
    main_met_obj.set_data(gt=gt, pred=pred, prob=prob)
    return main_met_obj

def get_ehr_adj(records, Nmed, no_weight=True, filter_th=None) -> np.array:
    ehr_adj = np.zeros((Nmed, Nmed))
    for patient in records:
        for adm in patient:
            med_set = adm[2]
            for i, med_i in enumerate(med_set):
                for j, med_j in enumerate(med_set):
                    if j<=i:
                        continue
                    ehr_adj[med_i, med_j] += 1
                    ehr_adj[med_j, med_i] += 1

    if filter_th is not None:
        ehr_adj[ehr_adj <= filter_th] = 0
    if no_weight:
        ehr_adj = ehr_adj.astype(bool).astype(int)

    return ehr_adj

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

# use the same metric from DMNC
def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def transform_split(X, Y):
    x_train, x_eval, y_train, y_eval = train_test_split(X, Y, train_size=2/3, random_state=1203)
    x_eval, x_test, y_eval, y_test = train_test_split(x_eval, y_eval, test_size=0.5, random_state=1203)
    return x_train, x_eval, x_test, y_train, y_eval, y_test

def sequence_output_process(output_logits, filter_token):
    pind = np.argsort(output_logits, axis=-1)[:, ::-1]

    out_list = []
    break_flag = False
    for i in range(len(pind)):
        if break_flag:
            break
        for j in range(pind.shape[1]):
            label = pind[i][j]
            if label in filter_token:
                break_flag = True
                break
            if label not in out_list:
                out_list.append(label)
                break
    y_pred_prob_tmp = []
    for idx, item in enumerate(out_list):
        y_pred_prob_tmp.append(output_logits[idx, item])
    sorted_predict = [x for _, x in sorted(zip(y_pred_prob_tmp, out_list), reverse=True)]
    return out_list, sorted_predict


def sequence_metric(y_gt, y_pred, y_prob, y_label):
    def average_prc(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b]==1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score


    def average_recall(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score


    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if (average_prc[idx] + average_recall[idx]) == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score


    def jaccard(y_gt, y_label):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = y_label[b]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_pred_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_pred_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob_label, k):
        precision = 0
        for i in range(len(y_gt)):
            TP = 0
            for j in y_prob_label[i][:k]:
                if y_gt[i, j] == 1:
                    TP += 1
            precision += TP / k
        return precision / len(y_gt)
    try:
        auc = roc_auc(y_gt, y_prob)
    except ValueError:
        auc = 0
    p_1 = precision_at_k(y_gt, y_label, k=1)
    p_3 = precision_at_k(y_gt, y_label, k=3)
    p_5 = precision_at_k(y_gt, y_label, k=5)
    f1 = f1(y_gt, y_pred)
    prauc = precision_auc(y_gt, y_prob)
    ja = jaccard(y_gt, y_label)
    avg_prc = average_prc(y_gt, y_label)
    avg_recall = average_recall(y_gt, y_label)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)


def multi_label_metric(y_gt, y_pred, y_prob):

    def jaccard(y_gt, y_pred):
        score = []
        # pdb.set_trace()
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            union = set(out_list) | set(target)
            jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def average_prc(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            prc_score = 0 if len(out_list) == 0 else len(inter) / len(out_list)
            score.append(prc_score)
        return score

    def average_recall(y_gt, y_pred):
        score = []
        for b in range(y_gt.shape[0]):
            target = np.where(y_gt[b] == 1)[0]
            out_list = np.where(y_pred[b] == 1)[0]
            inter = set(out_list) & set(target)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score.append(recall_score)
        return score

    def average_f1(average_prc, average_recall):
        score = []
        for idx in range(len(average_prc)):
            if average_prc[idx] + average_recall[idx] == 0:
                score.append(0)
            else:
                score.append(2*average_prc[idx]*average_recall[idx] / (average_prc[idx] + average_recall[idx]))
        return score

    def f1(y_gt, y_pred):
        all_micro = []
        for b in range(y_gt.shape[0]):
            all_micro.append(f1_score(y_gt[b], y_pred[b], average='macro'))
        return np.mean(all_micro)

    def roc_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(roc_auc_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_auc(y_gt, y_prob):
        all_micro = []
        for b in range(len(y_gt)):
            all_micro.append(average_precision_score(y_gt[b], y_prob[b], average='macro'))
        return np.mean(all_micro)

    def precision_at_k(y_gt, y_prob, k=3):
        precision = 0
        sort_index = np.argsort(y_prob, axis=-1)[:, ::-1][:, :k]
        for i in range(len(y_gt)):
            TP = 0
            for j in range(len(sort_index[i])):
                if y_gt[i, sort_index[i, j]] == 1:
                    TP += 1
            precision += TP / len(sort_index[i])
        return precision / len(y_gt)

    # roc_auc
    try:
        auc = roc_auc(y_gt, y_prob)
    except:
        auc = 0
    # precision
    p_1 = precision_at_k(y_gt, y_prob, k=1)
    p_3 = precision_at_k(y_gt, y_prob, k=3)
    p_5 = precision_at_k(y_gt, y_prob, k=5)
    # macro f1
    f1 = f1(y_gt, y_pred)
    # precision
    prauc = precision_auc(y_gt, y_prob)
    # jaccard
    ja = jaccard(y_gt, y_pred)
    # pre, recall, f1
    avg_prc = average_prc(y_gt, y_pred)
    avg_recall = average_recall(y_gt, y_pred)
    avg_f1 = average_f1(avg_prc, avg_recall)

    return ja, prauc, np.mean(avg_prc), np.mean(avg_recall), np.mean(avg_f1)

def dangerous_pair_num(record, t_record, path): #, ehr_path):
    # ddi rate
    if isinstance(path, str):
        ddi_A = dill.load(open(path, 'rb'))
    else:
        ddi_A = path
    '''
    if isinstance(ehr_path, str):
        ehr_A = dill.load(open(path, 'rb'))
    else:
        ehr_A = ehr_path
    '''
    all_cnt = 0
    dd_cnt = 0
    test_ddi_A = np.zeros_like(ddi_A)
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        # print((med_i, med_j))
                        # dd_cnt += 1
                        test_ddi_A[med_i, med_j] = 1
                        test_ddi_A[med_j, med_i] = 1
                        # if ehr_A[med_i, med_j] == 1 or ehr_A[med_j, med_i] == 1:
                            # dd_cnt -= 1
    for patient in t_record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    # all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        # print((med_i, med_j))
                        test_ddi_A[med_i, med_j] = 0
                        test_ddi_A[med_j, med_i] = 0                        
                        # dd_cnt += 1   
    if all_cnt == 0:
        return 0
    # return dd_cnt / all_cnt
    # return np.sum(test_ddi_A) / all_cnt
    print("in dangerous_pair_num(): ddi_rate is: ", np.sum(test_ddi_A) / all_cnt)
    return np.sum(test_ddi_A) # the number of the adverse DDI pairs
    # return dd_cnt

def ddi_rate_score(record, path):
    # ddi rate
    if isinstance(path, str):
        ddi_A = dill.load(open(path, 'rb'))
    else:
        ddi_A = path
    all_cnt = 0
    dd_cnt = 0
    # test_ddi_A = np.zeros_like(ddi_A)
    for patient in record:
        for adm in patient:
            med_code_set = adm
            for i, med_i in enumerate(med_code_set):
                for j, med_j in enumerate(med_code_set):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_A[med_i, med_j] == 1 or ddi_A[med_j, med_i] == 1:
                        # print((med_i, med_j))
                        dd_cnt += 1
                        # test_ddi_A[med_i, med_j] = 1
                        # test_ddi_A[med_j, med_i] = 1
                        # if ehr_A[med_i, med_j] == 1 or ehr_A[med_j, med_i] == 1:
                            # dd_cnt -= 1
    if all_cnt == 0:
        return 0
    return dd_cnt / all_cnt
    # return np.sum(test_ddi_A) / all_cnt
    # return np.sum(test_ddi_A)
    # return dd_cnt

class Metrics:
    def __init__(self, data, med_voc, args=None):
        self.med_voc = med_voc
        data = data
        cnts = Counter()
        self.args = args
        for p in data:
            for v in p:
                meds = v[-2]# meds = v[-1]
                cnts = cnts + Counter(meds)
        # divid the many, medium, few
        sorted_cnt = sorted(list(cnts.items()), key=lambda x: (-x[1], x[0]))
        self.medidx_ordered_desc, self.freqs_desc = list(zip(*sorted_cnt))
        Max_freq = sorted_cnt[0][1]
        th1, th2 = 0.6, 0.2
        th1, th2 = Max_freq*th1, Max_freq*th2
        self.stacks = defaultdict(list)
        i = 0
        while i < len(self.freqs_desc):
            if self.freqs_desc[i] <= 1000:
                break
            i += 1
        print("thre idx: {}".format(i))
        self.lowfreqmedidx = self.medidx_ordered_desc[0:30]
        
        ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final.pkl')

        self.ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
    
    def feed_data(self, y_gt, y_pred, y_prob):
        """
        patient level: (N_v, N_med)
        """
        for v_idx in range(len(y_gt)):
            cur_gt = y_gt[v_idx]
            cur_pred = y_pred[v_idx]
            cur_prob = y_prob[v_idx]
            self.stacks['gt'].append(cur_gt)
            self.stacks['pred'].append(cur_pred)
            self.stacks['prob'].append(cur_prob)

    def set_data(self, gt=None, pred=None, prob=None):
        if gt is None:
            self.gt = np.stack(self.stacks['gt'])
            self.pred = np.stack(self.stacks['pred'])
            self.prob = np.stack(self.stacks['prob'])
            self.stacks = defaultdict(list)
        else:
            self.gt, self.pred, self.prob = gt, pred, prob
        return

    def get_metric_res(self):
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
            self.gt, self.pred, self.prob)

        return ja, prauc, avg_p, avg_r, avg_f1
    
    def get_metric_res_for_cert_meds(self, meds:list, is_idx=True):
        meds = list(set(meds))
        if not is_idx:
            meds = [self.med_voc.word2idx[cur] for cur in meds]
        # pdb.set_trace()
        subgt, subpred, subprob = self.gt[:, meds], self.pred[:, meds], self.prob[:, meds]
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
            subgt, subpred, subprob)

        return ja, prauc, avg_p, avg_r, avg_f1

    def _jaccard(self, y_gt, y_pred):
        target = np.where(y_gt == 1)[0]
        out_list = np.where(y_pred == 1)[0]
        inter = set(out_list) & set(target)
        union = set(out_list) | set(target)
        jaccard_score = 0 if len(union) == 0 else len(inter) / len(union)
        return jaccard_score

    def get_metric_for_one_med(self, gt, pred, prob):
        ja = self._jaccard(gt, pred)
        f1 = f1_score(gt, pred)
        prauc = average_precision_score(gt, prob)
        return ja, f1, prauc

    def get_metric_by_freqs(self, idx="freq_desc", words=None):
        metrics = []
        for i in range(self.gt.shape[1]):
            cur_gt, cur_pred, cur_prob = self.gt[:, i], self.pred[:, i], self.prob[:, i]
            cur_res = self.get_metric_for_one_med(cur_gt, cur_pred, cur_prob)
            metrics.append(np.array(list(cur_res)))
        metrics = np.array(metrics)
        if idx == "freq_desc":
            freqordered_metrics = metrics[list(self.medidx_ordered_desc)]
            return freqordered_metrics
        elif idx == "norm":
            return  metrics
        elif idx == 'words':
            idx = [self.med_voc.word2idx[w] for w in words]
            return metrics[idx]
        return metrics

    def check_wrong(self, tar, cur):
        res = []
        for i in range(self.gt.shape[0]):
            if self.gt[i][tar] == 0 and self.gt[i][cur] == 0:
                continue
            elif self.gt[i][tar] * self.gt[i][cur] == 1:
                continue
            if self.gt[i][tar] == 1 and self.pred[i][tar] == 0 and self.pred[i][cur] == 1:
                # pdb.set_trace()
                gt_list = np.nonzero(self.gt[i])[0]
                pred_list = np.nonzero(self.pred[i])[0]
                res.append(i)
            if self.gt[i][cur] == 1 and self.pred[i][cur] == 0 and self.pred[i][tar] == 1:
                # pdb.set_trace()
                gt_list = np.nonzero(self.gt[i])[0]
                pred_list = np.nonzero(self.pred[i])[0]
                res.append(i)
        return res

    def run(self, ops="", **kwargs):
        ja, prauc, avg_p, avg_r, avg_f1 = self.get_metric_res()
        ddi_rate = -1  # if 'd' not in ops else 0.666
        if 'd' in ops:
            pred = self.pred
            gt = self.gt
            list_pred = []
            list_target = []
            for i in range(pred.shape[0]):
                idx = np.nonzero(pred[i])[0].tolist()
                list_pred.append(idx)
            for i in range(gt.shape[0]):
                idx = np.nonzero(gt[i])[0].tolist()
                list_target.append(idx)
            # ddi_rate = ddi_rate_score([list_pred], [list_target], self.ddi_adj, self.ehr_adj)
            ##print("in ddi_rate_score(), ddi_rate:", ddi_rate_score([list_pred], self.ddi_adj))
            ##ddi_rate = dangerous_pair_num([list_pred], [list_target], self.ddi_adj)#, self.ehr_adj) # 其实是DDI的对数（数据库中已有的不算）
            ddi_rate = ddi_rate_score([list_pred], self.ddi_adj)

        visit_cnt = self.gt.shape[0]
        med_cnt = self.pred.sum()
        print('adverse DDI number: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'.format(
            ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt
        ))
        if 'c' in ops:
            # idx = ['0054-3194', '0517-1010', '66213-423', '66213-425', '0225-0295', '66213-423', '66213-425']
            idx = ['0456-4020', '54838-540', '60505-2519', '0456-2010', '51079-543']
            c_ja, c_prauc, c_avg_p, c_avg_r, c_avg_f1 = self.get_metric_res_for_cert_meds(idx, False)
            print("---Certain meds metrics: ---")
            print('DDI Rate: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'.format(
                -1, np.mean(c_ja), np.mean(c_prauc), np.mean(c_avg_p), np.mean(c_avg_r), np.mean(c_avg_f1), -1
            ))
        if 'l' in ops:
            idx = self.lowfreqmedidx  # low frequency med indices or words
            c_ja, c_prauc, c_avg_p, c_avg_r, c_avg_f1 = self.get_metric_res_for_cert_meds(idx)
            print("---Certain meds metrics: ---")
            print('DDI Rate: {:.4f}, Jaccard: {:.4f},  PRAUC: {:.4f}, AVG_PRC: {:.4f}, AVG_RECALL: {:.4f}, AVG_F1: {:.4f}, AVG_MED: {:.4f}\n'.format(
                -1, np.mean(c_ja), np.mean(c_prauc), np.mean(c_avg_p), np.mean(c_avg_r), np.mean(c_avg_f1), -1
            ))
        if 'D' in ops:
            # topk precision
            topk = kwargs['topk']
            prob, gt = torch.Tensor(self.prob), torch.Tensor(self.gt)
            topk_val, topk_idx = torch.topk(prob, topk, dim=1)
            totol_label_num, hit_label_num = 0, 0
            # pdb.set_trace()
            for i in range(gt.shape[0]):
                cur_label = torch.nonzero(gt[i])[:, 0].tolist()
                totol_label_num += len(cur_label)
                cur_pred = topk_idx[i].tolist()
                cur_hit = [cur for cur in cur_pred if cur in set(cur_label)]
                hit_label_num += len(cur_hit)  
            acc = hit_label_num / totol_label_num
            print("topk-hit-acc: topk={}, acc={:.4f}".format(topk, acc))

        return ddi_rate, ja, prauc, avg_p, avg_r, avg_f1

