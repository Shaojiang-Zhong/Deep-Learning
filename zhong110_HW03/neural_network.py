import numpy as np
import math
import torch
class NeuralNetwork:
	def __init__(self, *args):
		self.layers = []
		self.Theta = {}
		self.a = {}
		self.z = {}
		self.dE_dTheta = {}
		self.forwardResult = 0
		self.lossValue = 0
		def generateWeight():
			for i in range(len(self.layers)-1):
				standardD = 1 / math.sqrt(self.layers[i])
				self.Theta[i] = torch.from_numpy(np.random.normal(0, standardD, (self.layers[i]+1, self.layers[i+1])))
		for arg in args:
			self.layers.append(arg)
		generateWeight()
	def getLayer(self, layer):
		return self.Theta[layer]
	def forward(self, input):
		if len(input.shape) == 1:
			#1d double tensor
			'''
			tmpResult = input
			print(tmpResult)
			for layer in range(len(self.layers)-1):
				print(tmpResult)
				# all weight stored in self.Theta[layer] matrice
				#tmpResult = np.dot(np.concatenate(np.asarray([1]), tmpResult, axis=0), self.Theta[layer])
				ones = torch.ones([1],dtype=torch.double)
				tmpTensor = torch.cat((ones, tmpResult))
				catTensor = (tmpTensor).unsqueeze(0)
				tmpResult = torch.mm(catTensor, self.Theta[layer])
				tmpResult = 1/(1+torch.exp(-tmpResult))
			return tmpResult
			'''
			tmpResult = torch.t(input.unsqueeze(0))
			for layer in range(len(self.layers)-1):
				# all weight stored in self.Theta[layer] matrice
				#tmpResult = np.dot(np.concatenate(np.ones(tmpResult.shape[0],1), tmpResult, axis=1), self.Theta[layer])
				#tmpResult = 1/(1+np.exp(-tmpResult))
				#print(tmpResult)
				ones = torch.ones([1, tmpResult.shape[1]],dtype=torch.double)
				#print("ones: " + str(ones))
				trans = torch.t(torch.cat((ones, tmpResult)))
				#print("trans: " + str(trans))
				self.a[layer] = torch.t(trans)
				tmpResult = torch.mm(trans, self.Theta[layer])
				#print("tmpResult: " + str(tmpResult))
				tmpResult = torch.t(tmpResult)
				#print("tmpResult: " + str(tmpResult))
				self.z[layer+1] = tmpResult
				tmpResult = 1/(1+torch.exp(-tmpResult))
			self.a[len(self.layers)-1] = tmpResult
			self.forwardResult = tmpResult[0]
			return tmpResult[0]

			pass
		elif len(input.shape) == 2:
			#2d double tensor
			#tmpResult = np.transpose(input.numpy())
			tmpResult = input
			for layer in range(len(self.layers)-1):
				# all weight stored in self.Theta[layer] matrice
				#tmpResult = np.dot(np.concatenate(np.ones(tmpResult.shape[0],1), tmpResult, axis=1), self.Theta[layer])
				#tmpResult = 1/(1+np.exp(-tmpResult))
				#print(tmpResult)
				ones = torch.ones([1, tmpResult.shape[1]],dtype=torch.double)
				trans = torch.t(torch.cat((ones, tmpResult)))
				self.a[layer] = torch.t(trans)
				tmpResult = torch.mm(trans, self.Theta[layer])
				tmpResult = torch.t(tmpResult)
				self.z[layer+1] = tmpResult
				tmpResult = 1/(1+torch.exp(-tmpResult))
			self.a[len(self.layers)-1] = tmpResult
			self.forwardResult = tmpResult
			return tmpResult
			pass
		else:
			print("error input")
			return []
	def backward(self, target):
		def lossCalculate(target):
			self.lossValue = (((self.forwardResult-target)**2).sum())/(2*len(target))
		lossCalculate(target)
		if len(target.shape) == 1:
			target = torch.t(target.unsqueeze(0))
		target = torch.t(target)
		diff_a = self.a[len(self.layers)-1] * (1 - self.a[len(self.layers)-1])
		tmpDelta = torch.mul((self.a[len(self.layers)-1] - target), diff_a)
		self.dE_dTheta[len(self.layers)-2] = torch.mm(self.a[len(self.layers)-2], torch.t(tmpDelta))
		#print("diff_a: "+ str(diff_a))
		#print("tmpDelta: " + str(tmpDelta))
		#print("self.dE_dTheta: "+str(self.dE_dTheta))
		for i in range(len(self.layers)-2, 0, -1):

			diff_a = self.a[i] * (1-self.a[i])
			#print("diff_a: "+ str(diff_a))
			x = self.Theta[i].mm(tmpDelta)
			#print("x: "+str(x))
			tmpDelta = torch.mul(x, diff_a) 
			#print("tmpDelta: " + str(tmpDelta))
			tmpDelta = tmpDelta[1:]
			#print("tmpDelta: " + str(tmpDelta))
			self.dE_dTheta[i-1] = torch.mm(self.a[i-1], tmpDelta.t())
			#print("self.dE_dTheta: "+str(self.dE_dTheta))
	def updateParams(self, eta):
		for i in range(len(self.layers)-1):
			self.Theta[i] -= self.dE_dTheta[i]*eta

