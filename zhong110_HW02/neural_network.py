import numpy as np
import math
import torch
class NeuralNetwork:
	def __init__(self, *args):
		self.layers = []
		self.theta = {}
		def generateWeight():
			for i in range(len(self.layers)-1):
				standardD = 1 / math.sqrt(self.layers[i])
				self.theta[i] = torch.from_numpy(np.random.normal(0, standardD, (self.layers[i]+1, self.layers[i+1])))
		for arg in args:
			self.layers.append(arg)
		generateWeight()
	def getLayer(self, layer):
		return self.theta[layer]
	def forward(self, input):
		if len(input.shape) == 1:
			#1d double tensor
			'''
			tmpResult = input
			print(tmpResult)
			for layer in range(len(self.layers)-1):
				print(tmpResult)
				# all weight stored in self.theta[layer] matrice
				#tmpResult = np.dot(np.concatenate(np.asarray([1]), tmpResult, axis=0), self.theta[layer])
				ones = torch.ones([1],dtype=torch.double)
				tmpTensor = torch.cat((ones, tmpResult))
				catTensor = (tmpTensor).unsqueeze(0)
				tmpResult = torch.mm(catTensor, self.theta[layer])
				tmpResult = 1/(1+torch.exp(-tmpResult))
			return tmpResult
			'''
			tmpResult = torch.t(input.unsqueeze(0))
			for layer in range(len(self.layers)-1):
				# all weight stored in self.theta[layer] matrice
				#tmpResult = np.dot(np.concatenate(np.ones(tmpResult.shape[0],1), tmpResult, axis=1), self.theta[layer])
				#tmpResult = 1/(1+np.exp(-tmpResult))
				#print(tmpResult)
				ones = torch.ones([1, tmpResult.shape[1]],dtype=torch.double)
				trans = torch.t(torch.cat((ones, tmpResult)))
				tmpResult = torch.mm(trans, self.theta[layer])
				tmpResult = 1/(1+torch.exp(-tmpResult))
				tmpResult = torch.t(tmpResult)
			return tmpResult[0]

			pass
		elif len(input.shape) == 2:
			#2d double tensor
			#tmpResult = np.transpose(input.numpy())
			tmpResult = input
			for layer in range(len(self.layers)-1):
				# all weight stored in self.theta[layer] matrice
				#tmpResult = np.dot(np.concatenate(np.ones(tmpResult.shape[0],1), tmpResult, axis=1), self.theta[layer])
				#tmpResult = 1/(1+np.exp(-tmpResult))
				#print(tmpResult)
				ones = torch.ones([1, tmpResult.shape[1]],dtype=torch.double)
				trans = torch.t(torch.cat((ones, tmpResult)))
				tmpResult = torch.mm(trans, self.theta[layer])
				tmpResult = 1/(1+torch.exp(-tmpResult))
				tmpResult = torch.t(tmpResult)
			return tmpResult
			pass
		else:
			print("error input")
			return []
