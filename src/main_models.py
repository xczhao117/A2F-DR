import os
import dill
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pdb
from torch.nn.parameter import Parameter

from pretrain_heads import FragNetPreTrain

from datetime import datetime


class main_model(nn.Module):
	def __init__(self, vocab_size, ddi_adj,
				 emb_dim=256,
				 device=torch.device('cpu:0'),
				 args=None):
		super().__init__()
		self.args = args
		self.tensor_ddi_adj = torch.FloatTensor(ddi_adj).to(device)		
		self.device = device
		self.vocab_size = vocab_size

		# pre-embedding
		self.embeddings = nn.ModuleList(
			[nn.Embedding(vocab_size[i]+1, emb_dim, padding_idx=vocab_size[i]) for i in range(2)])
		self.dropout = nn.Dropout(p=0.5)
		self.encoders = nn.ModuleList([nn.GRU(emb_dim, emb_dim, batch_first=True) for _ in range(2)])
		self.query = nn.Sequential(
				nn.ReLU(),
				nn.Linear(2 * emb_dim, emb_dim)
		)

		if args.use_mol_net:
			self.final_map = nn.Sequential(
					nn.Linear(2 * emb_dim, emb_dim),
					nn.ReLU(),
					nn.Linear(emb_dim, self.vocab_size[2]),
					nn.LayerNorm(self.vocab_size[2])
			)
			self.fragnet = FragNetPreTrain(num_layer=4,
										drop_ratio=0.2,
										num_heads=4,
										emb_dim=128,
										atom_features=167,
										frag_features=167,
										edge_features=17,
										fedge_in=6,
										fbond_edge_in=6,
										nt=args.mol_net_type)
		else:
			self.final_map = nn.Sequential(
					nn.Linear(emb_dim, emb_dim),
					nn.ReLU(),
					nn.Linear(emb_dim, self.vocab_size[2]),
					nn.LayerNorm(self.vocab_size[2])
			)
		if args.use_mol_net and args.use_mol_loss:
			self.frag_loss_fn = nn.MSELoss()

	def get_inputs(self, dataset, MaxVisit=2):
		# 将list的数据形式转换为tensor形式的
		# use the pad index to make th same length tensor
		diag_list, pro_list, med_list, med_ml_list, len_list = [], [], [], [], []
		all_mol_list = []
		max_visit = min(max([len(cur) for cur in dataset]), MaxVisit) # finnaly max_visit=2
		ml_diag = max([len(dataset[i][j][0]) for i in range(len(dataset)) for j in range(len(dataset[i]))]) # max number of diagnoses in one visit
		ml_pro = max([len(dataset[i][j][1]) for i in range(len(dataset)) for j in range(len(dataset[i]))]) # max number of procedures in one visit
		# [v1, v2, v3] -> [v1], [v1, v2], [v2, v3]
		
		zxc_med_cnt = 0
		zxc_smp_cnt = 0
		
		for p in dataset:
			cur_diag = torch.full((max_visit, ml_diag), self.vocab_size[0])
			cur_pro = torch.full((max_visit, ml_pro), self.vocab_size[1])
			cur_mols = []
			for ad_idx in range(len(p)):
				d_list, p_list, m_list, mol_list = p[ad_idx]
				if ad_idx >= max_visit:
					cur_diag[:-1] = cur_diag[1:]
					cur_pro[:-1] = cur_pro[1:]
					cur_diag[-1] = self.vocab_size[0]
					cur_pro[-1] = self.vocab_size[1]
					cur_diag[-1, :len(d_list)] = torch.LongTensor(d_list)
					cur_pro[-1, :len(p_list)] = torch.LongTensor(p_list)
					# visit len mask
					len_list.append(max_visit)
				else:
					cur_diag[ad_idx, :len(d_list)] = torch.LongTensor(d_list) 
					cur_pro[ad_idx, :len(p_list)] = torch.LongTensor(p_list)
					# visit len mask
					len_list.append(ad_idx + 1)


				cur_mols.append(mol_list)
				real_cur_mol = []
				if ad_idx == 0:
					real_cur_mol = cur_mols[0]
				else:
					real_cur_mol = cur_mols[ad_idx - 1]

				all_mol_list.append(real_cur_mol)

				diag_list.append(cur_diag.long().clone())
				pro_list.append(cur_pro.long().clone())
				# bce target
				cur_med = torch.zeros(self.vocab_size[2])
				cur_med[m_list] = 1
				med_list.append(cur_med)
				# multi-label margin target
				cur_med_ml = torch.full((self.vocab_size[2],), -1)
				cur_med_ml[:len(m_list)] = torch.LongTensor(m_list)
				med_ml_list.append(cur_med_ml)
				zxc_med_cnt += len(m_list)
				zxc_smp_cnt += 1
		print('avg. Drug Number: ', zxc_med_cnt/zxc_smp_cnt)

		diag_tensor = torch.stack(diag_list)
		pro_tensor = torch.stack(pro_list)
		med_tensor_bce_target = torch.stack(med_list)
		med_tensor_ml_target = torch.stack(med_ml_list)
		len_tensor = torch.LongTensor(len_list)
	
		return diag_tensor, pro_tensor, med_tensor_bce_target, med_tensor_ml_target, len_tensor, all_mol_list
	
	def get_batch(self, data, batchsize=None):
		# diag_tensor, pro_tensor, med_tensor, len_tensor, mol_tensor
		# data = self.get_inputs(dataset)
		if batchsize is None:
			yield data
		else:
			N = data[0].shape[0]
			#idx = np.arange(N).astype(int)
			idx = [j for j in range(N)]
			np.random.shuffle(idx)
			i = 0
			while i < N:
				cur_idx = idx[i:i+batchsize]
				
				cur_diag = data[0][cur_idx]
				cur_pro = data[1][cur_idx]
				cur_med_bce = data[2][cur_idx]
				cur_med_ml = data[3][cur_idx]
				cur_len = data[4][cur_idx]
				# mol_list
				cur_mol = []
				for j in cur_idx:
					cur_mol.append(data[5][j])

				res = cur_diag, cur_pro, cur_med_bce, cur_med_ml, cur_len, cur_mol
				
				yield res
				i += batchsize

	def _get_query(self, diag, pro, visit_len):
		diag_emb_seq = self.dropout(self.embeddings[0](diag).sum(-2))
		pro_emb_seq = self.dropout(self.embeddings[1](pro).sum(-2))
		o1, h1 = self.encoders[0](diag_emb_seq)
		o2, h2 = self.encoders[1](pro_emb_seq)  # o2 with shape (B, M, D)
		# NOTE: select by len
		# o1, o2 with shape (B, D)
		o1 = torch.stack([o1[i,visit_len[i]-1, :] for i in range(visit_len.shape[0])]) #每个batch挑一个o1或者o2
		o2 = torch.stack([o2[i,visit_len[i]-1, :] for i in range(visit_len.shape[0])])

		patient_representations = torch.cat([o1, o2], dim=-1)  # (B, dim*2)
		query = self.query(patient_representations)  # (B, dim)

		norm_of_query = torch.norm(query, 2, 1, keepdim=True)
		normed_query = (norm_of_query / (1 + norm_of_query)) * (query / norm_of_query)	   
		return query, normed_query

	def forward(self, input, visit_len):
		diag, pro, labels, mols, mols_len = input # labels are never used
		query, normed_query = self._get_query(diag, pro, visit_len)  # (Batch, dim)

		if self.args.use_mol_net:
			bond_length_pred, bond_angle_pred, dihedral_angle_pred, graph_rep1, graph_rep2 = self.fragnet(mols)

		if self.args.use_mol_net and self.args.use_mol_loss:
			bond_length_true = mols['bnd_lngth']
			bond_angle_true = mols['bnd_angl']
			dihedral_angle_true = mols['dh_angl']
			E = mols['y']

			loss_lngth = self.frag_loss_fn(bond_length_pred, bond_length_true)
			loss_angle = self.frag_loss_fn(bond_angle_pred, bond_angle_true)
			loss_lngth = self.frag_loss_fn(dihedral_angle_pred, dihedral_angle_true)
			loss_E = self.frag_loss_fn(graph_rep1.view(-1), E)
			##
			frag_loss = loss_lngth + loss_angle + loss_lngth + loss_E
		else:
			frag_loss = 0

		# split graph_rep2 ()
		if self.args.use_mol_net:
			start_idx = 0;
			mol_fea = []
			for curr_len in mols_len:
				end_idx = start_idx + curr_len
				curr_gr = torch.mean(graph_rep2[start_idx:end_idx], dim=0)
				mol_fea.append(curr_gr)
				start_idx = end_idx
			mol_fea_tensor = torch.stack(mol_fea) # should be (Batch, dim)

			normed_mol_fea_tensor = mol_fea_tensor / torch.norm(mol_fea_tensor, 2, 1, keepdim=True)

			final_rep = torch.cat([normed_query, normed_mol_fea_tensor], dim=-1)  # (B, dim*2)
		else:
			final_rep = normed_query
		
		result = self.final_map(final_rep) # prediction of medcines

		if self.args.ddi: 
			neg_pred_prob = F.sigmoid(result)
			tmp_left = neg_pred_prob.unsqueeze(2)  # (B, Nmed, 1)
			tmp_right = neg_pred_prob.unsqueeze(1)  # (B, 1, Nmed)
			neg_pred_prob = torch.matmul(tmp_left, tmp_right)  # (N, Nmed, Nmed)
			batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum() # ddi loss
		else:
			batch_neg = 0   # ddi_loss

		return result, batch_neg, frag_loss
	
