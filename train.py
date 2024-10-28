import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import sys
import os
import torch
import torch.nn.functional as F

from models import CasualTPP


REG = 0.1


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def load_data(K, tau):
	'''
	Input:
	K: length of seq feature
	tau: length of target ahead

	Returns:
	seq_feature (K, fea_dim): including the treatment
	group_label: the sample belongs to treatment group or control group
	outcome_label: the target outcome
	'''
	data = pd.read_csv('data/simulation/simulate_chatroom_1.csv')
	dataLen = len(data)
	# past treatments + other features
	seq_feature_columns = ['current_moderation_count', 'total_badges_messages', 'ViewerNet_Connectivity']
	treatment_column = 'current_moderation_count'
	outcome_column = 'total_badges_messages'
	featureDim = len(seq_feature_columns)

	# construct sample
	seq_fea_list, treat_label_list, y_label_list = [], [], []
	seq_fea_cf_list = []
	for i in range(dataLen-K-tau):
		seq_fea_i = data[seq_feature_columns][i:i+K]
		# creating counterfactual features
		seq_fea_cf_i = seq_fea_i.copy()
		seq_fea_cf_i.iloc[-1,seq_fea_cf_i.columns.get_loc(treatment_column)] = 1-seq_fea_cf_i.iloc[-1,seq_fea_cf_i.columns.get_loc(treatment_column)]

		treatment_i = data[treatment_column][i+K-1]
		outcome_i = data[outcome_column][i+K] # tau=1
		seq_fea_list.append(seq_fea_i)
		treat_label_list.append(treatment_i)
		y_label_list.append(outcome_i)

		seq_fea_cf_list.append(seq_fea_cf_i)
	return np.array(seq_fea_list), np.array(seq_fea_cf_list), np.array(treat_label_list), np.array(y_label_list), featureDim



def train():
	seq_feature, seq_feature_cf, treat_label, outcome_label, featureDim = load_data(K=10, tau=1)
	print("Check data shape: ", seq_feature.shape, treat_label.shape, outcome_label.shape)

	## TODO: batch training

	model = CasualTPP(in_dim=featureDim)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
	for epoch in range(100):
		model.train()

		optimizer.zero_grad()
		treat_pred, outcome_pred = model(torch.FloatTensor(seq_feature))
		treat_loss = torch.nn.CrossEntropyLoss()(treat_pred, torch.LongTensor(treat_label))
		out_loss = F.mse_loss(outcome_pred.float(), torch.FloatTensor(outcome_label))
		loss = out_loss + REG*treat_loss
		loss.backward()
		optimizer.step()

		if epoch%10 == 0:
			model.eval()
			treat_id, control_id = np.nonzero(treat_label)[0], np.nonzero(1-treat_label)[0]
			print("Num treat/control: ", len(treat_id), len(control_id))

			_, pred_treat_y = model(torch.FloatTensor(seq_feature)[treat_id])
			_, pred_treat_y_cf = model(torch.FloatTensor(seq_feature_cf)[control_id])

			_, pred_control_y = model(torch.FloatTensor(seq_feature)[control_id])
			_, pred_control_y_cf = model(torch.FloatTensor(seq_feature_cf)[treat_id])


			y1_hat = torch.cat([pred_treat_y, pred_treat_y_cf],dim=0).squeeze(-1).detach().numpy()
			y0_hat = torch.cat([pred_control_y_cf, pred_control_y],dim=0).squeeze(-1).detach().numpy()

			print("treat ave: ", np.mean(y1_hat), "control ave: ", np.mean(y0_hat))
			print("ATE: ", np.mean(y1_hat)-np.mean(y0_hat))


if __name__ == "__main__":
	seed = 101
	set_seed(seed)
	#load_data(10, 1)
	train()