import sys
import time
import math
import dill
import numpy as np
import argparse
from collections import defaultdict
from sklearn import metrics
from sklearn.metrics import jaccard_score
from torch.optim import Adam, SGD
import os
import torch
from main_models import main_model
from util import ddi_rate_score, Metrics
import torch.nn.functional as F
from fragnet.dataset.data import collate_fn_pt as collate_fn # process a list of Data into a dict object for training

from datetime import datetime


#1) EHR + acc
#python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_ACC_a0.95

#2) EHR + acc + ddi
#python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_ACC_a0.95_DDI_g0.9 --ddi

#3) EHR + MOL-atom + acc
#python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_MOLa_ACC_a0.95 --use_mol_net --mol_net_type 2

#4) EHR + MOL-atom + acc + ddi
#python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_MOLa_ACC_a0.95_DDI_g0.9 --use_mol_net --mol_net_type 2 --ddi

#5) EHR + MOL-frag + acc
#python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_MOLf_ACC_a0.95 --use_mol_net --mol_net_type 3

#6) EHR + MOL-frag + acc + ddi (A2F-DR(\beta=0.9))
#python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_MOLf_ACC_a0.95_DDI_g0.9 --use_mol_net --mol_net_type 3 --ddi

#7) EHR + MOL-all + acc
#python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_MOL_ACC_a0.95 --use_mol_net --mol_net_type 1

#8) EHR + MOL-all + acc + ddi
#python ./src/run_mimic_iii.py --cuda 0 --epoch 120 --model_name EHR_MOL_ACC_a0.95_DDI_g0.9 --use_mol_net --mol_net_type 1 --ddi

parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default='model_name', help="model name")
parser.add_argument('--resume_path', type=str, default='resume_path', help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--datadir', type=str, default="./data/processed/", help='datadir')
parser.add_argument('--cuda', type=int, default=-1, help='use cuda')
parser.add_argument('--seed', type=int, default=1029, help='random seed')
parser.add_argument('--epoch', type=int, default=114, help='# of epoches')
parser.add_argument('--load', action='store_true', default=False, help='load resume file')
parser.add_argument('--ddi', action='store_true', default=False, help='use ddi')
parser.add_argument('--target_ddi', type=float, default=0.005, help='target ddi')

parser.add_argument('--alpha', type=float, default=0.95, help='weight for loss_bce and loss_multi')
parser.add_argument('--beta',  type=float, default=1, help='weight for loss_cls and loss_frag')
parser.add_argument('--gamma', type=float, default=0.9, help='weight for loss_acc and loss_ddi')

parser.add_argument('--use_mol_net', action='store_true', default=False, help='use the molecule encoding sub-network')
parser.add_argument('--mol_net_type', type=int, default=3, help='1--full, 2--atom only, 3--frag only')

parser.add_argument('--use_mol_loss', action='store_true', default=False, help='use the mol_loss (--use_mol_net must be true)') # always false


args = parser.parse_args()
print(args)

if not os.path.exists(os.path.join("./src/saved_mimic3", args.model_name)):
		os.makedirs(os.path.join("./src/saved_mimic3", args.model_name))

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.cuda > -1:
	torch.cuda.manual_seed(args.seed)


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def eval(model: main_model, data_eval, voc_size, epoch, metric_obj: Metrics):
	model.eval()

	ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
	med_cnt, visit_cnt = 0, 0

	for data_tensors in model.get_batch(data_eval, 16):
		cur_diag, cur_pro, cur_med_target, _, cur_len, cur_mol = data_tensors

		cur_diag = cur_diag.to(model.device)
		cur_pro = cur_pro.to(model.device)
		cur_med_target = cur_med_target.to(model.device)
		cur_len = cur_len.to(model.device)

		#把所有的mol当作一个batch算出来，loss一起算，但是要把graph_rep2进行拆分，从而得到对应一个病人的药物信息
		#   并且分别映射为一个向量，用于与normed_query拼接
		mol_batch = [single_mol for sublist in cur_mol for single_mol in sublist]
		# fragnet种的一个data point是一个分子的信息。collate_fn将所有的分子处理为一个dict object才能被fragnet处理
		mol_batch_dict = collate_fn(mol_batch)
		for k,v in mol_batch_dict.items():
			mol_batch_dict[k] = mol_batch_dict[k].to(model.device)

		# 拆分graph_rep2 需要知道每个样本对应几个分子
		mols_len = []
		for sublist in cur_mol:
			mols_len.append(len(sublist))

		result, _, _ = model((cur_diag, cur_pro, None, mol_batch_dict, mols_len), cur_len)

		result = F.sigmoid(result).detach().cpu().numpy()
		preds = np.zeros_like(result)
		preds[result>=0.5] = 1
		preds[result<0.5] = 0
		visit_cnt += cur_med_target.shape[0]
		med_cnt += preds.sum()
		cur_med_target = cur_med_target.detach().cpu().numpy()
		metric_obj.feed_data(cur_med_target, preds, result)

	ddi_rate = -1
	metric_obj.set_data()
	ops = 'd'
	ddi_rate, ja, prauc, avg_p, avg_r, avg_f1 = metric_obj.run(ops=ops)
	return ddi_rate, np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), med_cnt / visit_cnt



def main():
	data_path = os.path.join(args.datadir, 'records_final.pkl')
	voc_path = os.path.join(args.datadir, 'voc_final.pkl')
	ddi_adj_path = os.path.join(args.datadir, 'ddi_A_final.pkl')

	device = torch.device('cuda:'+str(args.cuda) if args.cuda > -1 else 'cpu')

	ddi_adj = dill.load(open(ddi_adj_path, 'rb'))
	data = dill.load(open(data_path, 'rb'))
	voc = dill.load(open(voc_path, 'rb'))
	
	diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
	voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

	metric_obj = Metrics(data, med_voc, 'ddi_A_final.pkl', args)

	split_point = int(len(data) * 2 / 3)
	data_train = data[:split_point]
	eval_len = int(len(data[split_point:]) / 2)
	data_test = data[split_point:split_point + eval_len]
	data_eval = data[split_point+eval_len:]

	model = main_model(voc_size, ddi_adj, 
						emb_dim=args.dim,
						device=device,
						args=args)
	model.to(device=device)
	optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	# optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.95, weight_decay=1e-5)
	epoch_begin = 0

	args.resume_path = f'./src/saved_mimic3/{args.model_name}/best.model'

	if args.Test or args.load:
		checkpoint = torch.load(args.resume_path, map_location=device)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		epoch_begin = checkpoint['epoch'] + 1
		print(f"Load {args.resume_path} finish...")

	if args.Test:
		model.to(device=device)
		tic = time.time()
		print('--------------------Begin Testing--------------------')
		ddi_list, ja_list, prauc_list, f1_list, med_list = [], [], [], [], []
		tic, result, sample_size = time.time(), [], round(len(data_test) * 0.8)
		np.random.seed(0)
		for _ in range(10):
			# test_sample = np.random.choice(data_test, sample_size, replace=True)
			test_sample_indices = np.random.choice(len(data_test), sample_size, replace=True)
			test_sample = [data_test[i] for i in test_sample_indices]
			test_sample_tensors = model.get_inputs(test_sample)
			ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, test_sample_tensors, voc_size, 0, metric_obj) 
			result.append([ddi_rate, ja, avg_f1, prauc, avg_med])
		result = np.array(result)
		mean, std = result.mean(axis=0), result.std(axis=0)
		metric_list = ['ddi_rate', 'ja', 'avg_f1', 'prauc', 'med']
		outstring = ''.join([
			"{}:\t{:.4f} $\\pm$ {:.4f} & \n".format(metric_list[idx], m, s)
			for idx, (m, s) in enumerate(zip(mean, std))
		])
		print(outstring)
		#'''
		print ('test time: {}'.format(time.time() - tic))
		return 

	# start iterations
	history = defaultdict(list)
	best_epoch, best_ja = 0, 0

	# 将list数据转化为tensor
	data_train_tensors = model.get_inputs(data_train)
	data_eval_tensors = model.get_inputs(data_eval)
	EPOCH = args.epoch
	for epoch in range(epoch_begin, EPOCH):
		tic = time.time()
		print ('\nepoch {} --------------------------'.format(epoch))
		
		model.train()
		step = 0
		trian_visit_num = sum([len(p) for p in data_train])
		
		epoch_loss_ddi = []
		epoch_loss_frag = []
		epoch_loss_bce = []
		epoch_loss_multi = []
		epoch_loss_total = []

		for cur_batch in model.get_batch(data_train_tensors, 16):
			cur_diag, cur_pro, cur_med_bce_target, cur_med_ml_target, cur_len, cur_mol = cur_batch

			# cur_diag, cur_pro, cur_med_target, _, cur_len, cur_mol = cur_batch
			cur_diag = cur_diag.to(device)
			cur_pro = cur_pro.to(device)
			cur_med_bce_target = cur_med_bce_target.to(device)
			cur_med_ml_target = cur_med_ml_target.to(device)
			cur_len = cur_len.to(device)

			#把所有的mol当作一个batch算出来，loss一起算，但是要把graph_rep2进行拆分，从而得到对应一个病人的药物信息
			#   并且分别映射为一个向量，用于与normed_query拼接
			mol_batch = [single_mol for sublist in cur_mol for single_mol in sublist]
			# fragnet种的一个data point是一个分子的信息。collate_fn将所有的分子处理为一个dict object才能被fragnet处理
			mol_batch_dict = collate_fn(mol_batch)
			for k,v in mol_batch_dict.items():
				mol_batch_dict[k] = mol_batch_dict[k].to(device)
			
			# 拆分graph_rep2 需要知道每个样本对应几个分子
			mols_len = []
			for sublist in cur_mol:
				mols_len.append(len(sublist))

			##-----------##
			result, loss_ddi, loss_frag = model((cur_diag, cur_pro, cur_med_bce_target, mol_batch_dict, mols_len), cur_len)

			# NOTE: batch of these loss function
			loss_bce = F.binary_cross_entropy_with_logits(result, cur_med_bce_target)
			loss_multi = F.multilabel_margin_loss(F.sigmoid(result), cur_med_ml_target)

			# losses
			loss_acc = args.alpha * loss_bce + (1 - args.alpha) * loss_multi
			#
			if args.use_mol_net and args.use_mol_loss:
				loss_pred = args.beta * loss_acc + (1 - args.beta) * loss_frag
			else:
				loss_pred = loss_acc
			#
			if args.ddi:
				labellist = []
				for i in range(cur_med_ml_target.shape[0]):
					cur = torch.nonzero(cur_med_ml_target[i])[:, 0].tolist()
					labellist.append(cur)
				cur_ddi_rate = ddi_rate_score([labellist], ddi_adj)
				if cur_ddi_rate > args.target_ddi:   # 如果当前ddi率大于目标ddi率，则加入ddi loss
					loss = args.gamma * loss_pred + (1 - args.gamma) * loss_ddi
			else:
				loss = loss_pred

			epoch_loss_ddi.append(loss_ddi)
			epoch_loss_frag.append(loss_frag)
			epoch_loss_bce.append(loss_bce)
			epoch_loss_multi.append(loss_multi)
			epoch_loss_total.append(loss)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			step += cur_diag.shape[0]
			llprint('\rtraining step: {} / {}'.format(step, trian_visit_num))
		#'''
		print()
		tic2 = time.time() 
		ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_eval_tensors, voc_size, epoch, metric_obj)
		print ('training time: {}, test time: {}'.format(tic2 - tic, time.time() - tic2))

		history['ja'].append(ja)
		history['ddi_rate'].append(ddi_rate)
		history['avg_p'].append(avg_p)
		history['avg_r'].append(avg_r)
		history['avg_f1'].append(avg_f1)
		history['prauc'].append(prauc)
		history['med'].append(avg_med)

		history['loss_ddi'].append(sum(epoch_loss_ddi)/len(epoch_loss_ddi))
		history['loss_frag'].append(sum(epoch_loss_frag)/len(epoch_loss_frag))
		history['loss_bce'].append(sum(epoch_loss_bce)/len(epoch_loss_bce))
		history['loss_multi'].append(sum(epoch_loss_multi)/len(epoch_loss_multi))
		history['loss_total'].append(sum(epoch_loss_total)/len(epoch_loss_total))

		if epoch >= 5:
			print ('ddi: {}, Med: {}, Ja: {}, F1: {}, PRAUC: {}'.format(
				np.mean(history['ddi_rate'][-5:]),
				np.mean(history['med'][-5:]),
				np.mean(history['ja'][-5:]),
				np.mean(history['avg_f1'][-5:]),
				np.mean(history['prauc'][-5:])
				))
		#'''
		savefile = os.path.join('./src/saved_mimic3', args.model_name, 'Epoch_%d_JA_%.4f_DDI_%.4f.model' % (epoch, 0, 0))
		torch.save({"model": model.state_dict(),
					"optimizer": optimizer.state_dict(),
					"epoch": epoch}, open(savefile, 'wb'))
		#'''
		if best_ja < ja:
			best_epoch = epoch
			best_ja = ja
			savefile = os.path.join('./src/saved_mimic3', args.model_name, 'best.model')
			torch.save({"model": model.state_dict(),
						"optimizer": optimizer.state_dict(),
						"epoch": epoch}, open(savefile, 'wb'))

		print ('best_epoch: {}'.format(best_epoch))

	dill.dump(history, open(os.path.join('./src/saved_mimic3', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))


if __name__ == '__main__':
	main()
