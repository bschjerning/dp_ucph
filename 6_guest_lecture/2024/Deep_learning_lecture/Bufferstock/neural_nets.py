import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
	""" Policy function """
	
	def __init__(self,state_dim,action_dim,Nneurons,final_activation="sigmoid"):

		super(Policy, self).__init__()
		
		self.layers =  nn.ModuleList([None]*(Nneurons.size+1))
	
		# input layer
		self.layers[0] = nn.Linear(state_dim, Nneurons[0])

		# hidden layers
		for i in range(1,len(self.layers)-1):
			self.layers[i] = nn.Linear(Nneurons[i-1], Nneurons[i])
		
		# output layer
		self.layers[-1] = nn.Linear(Nneurons[-1], action_dim)

		if hasattr(F, final_activation):
			self.final_activation = getattr(F,final_activation)
		else:
			raise ValueError(f'final_activation {final_activation} function not availible')

	def forward(self, state):
		""" Forward pass"""

        # input layer
		s = F.relu(self.layers[0](state))
	
        # hidden layers
		for i in range(1,len(self.layers)-1):
			s = F.relu(self.layers[i](s))

		# output layer
		return self.final_activation(self.layers[-1](s))