import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.seq2seq import Seq2Seq, EncoderRNN
from models.social_stgcn import st_gcn
import numpy as np 

class GRIPModel(nn.Module):
	def __init__(self, in_channels, num_node, edge_importance_weighting, **kwargs):
		super().__init__()

		# load graph
		#self.graph = Graph(**graph_args)
		max_hops=2
		A = np.ones((max_hops+1, num_node, num_node))

		# build networks
		spatial_kernel_size = np.shape(A)[0]
		temporal_kernel_size = 5 #9 #5 # 3
		kernel_size = (temporal_kernel_size, spatial_kernel_size)

		# best
		self.st_gcn_networks = nn.ModuleList((
			nn.BatchNorm2d(in_channels),
			st_gcn(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
			st_gcn(64, 64, kernel_size, 1, **kwargs),
			st_gcn(64, 64, kernel_size, 1, **kwargs),
		))

		# initialize parameters for edge importance weighting
		if edge_importance_weighting:
			self.edge_importance = nn.ParameterList(
				[nn.Parameter(torch.ones(np.shape(A))) for i in self.st_gcn_networks]
				)
		else:
			self.edge_importance = [1] * len(self.st_gcn_networks)

		self.num_node = num_node
		self.out_dim_per_node = out_dim_per_node = 2 #(x, y) coordinate
		self.seq2seq_car = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5)


	def reshape_for_lstm(self, feature):
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, T, V = feature.size() 
		now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
		now_feat = now_feat.view(N*V, T, C) 
		return now_feat

	def reshape_from_lstm(self, predicted):
		# predicted (N*V, T, C)
		NV, T, C = predicted.size()
		now_feat = predicted.view(-1, self.num_node, T, self.out_dim_per_node) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
		now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
		return now_feat

	def forward(self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
		x = pra_x
		
		# forwad
		for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
			if type(gcn) is nn.BatchNorm2d:
				x = gcn(x)
			else:
				x, _ = gcn(x, pra_A + importance)
				
		# prepare for seq2seq lstm model
		graph_conv_feature = self.reshape_for_lstm(x)
		last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]

		if pra_teacher_forcing_ratio>0 and type(pra_teacher_location) is not type(None):
			pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

		# now_predict.shape = (N, T, V*C)
		now_predict = self.seq2seq_car(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length)
		now_predict = self.reshape_from_lstm(now_predict) # (N, C, T, V)

		return now_predict 

if __name__ == '__main__':
	model = Model(in_channels=4, pred_length=3, graph_args={}, edge_importance_weighting=False)
	print(model)