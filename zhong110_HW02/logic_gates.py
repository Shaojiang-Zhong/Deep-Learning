from neural_network import NeuralNetwork
import torch
class AND:
	def __init__(self):
		self.nn = NeuralNetwork(2,1)
		#print(self.nn.theta)
		self.thetaTmp = self.nn.getLayer(0)
		self.thetaTmp.fill_(0)
		self.thetaTmp += torch.tensor([[-1.5], [1], [1]], dtype=torch.double)
	def __call__(self, x, y):
		self.x = x
		self.y = y
		result = self.forward()
		if result > 0.5: return True
		else: return False
	def forward(self):
		return self.nn.forward(torch.tensor([self.x, self.y],dtype=torch.double))




class OR:
	def __init__(self):
		self.nn = NeuralNetwork(2,1)
		#print(self.nn.theta)
		self.thetaTmp = self.nn.getLayer(0)
		self.thetaTmp.fill_(0)
		self.thetaTmp += torch.tensor([[-0.5], [1], [1]], dtype=torch.double)
	def __call__(self, x, y):
		self.x = x
		self.y = y
		result = self.forward()
		if result > 0.5: return True
		else: return False
	def forward(self):
		return self.nn.forward(torch.tensor([self.x, self.y],dtype=torch.double))
class NOT:
	def __init__(self):
		self.nn = NeuralNetwork(1,1)
		#print(self.nn.theta)
		self.thetaTmp = self.nn.getLayer(0)
		self.thetaTmp.fill_(0)
		self.thetaTmp += torch.tensor([[0.5], [-1]], dtype=torch.double)
	def __call__(self, x):
		self.x = x
		result = self.forward()
		if result > 0.5: return True
		else: return False
	def forward(self):
		return self.nn.forward(torch.tensor([self.x],dtype=torch.double))

class XOR:
	def __init__(self):
		self.nn = NeuralNetwork(2,2,1)
		#print(self.nn.theta)
		self.thetaTmp = self.nn.getLayer(0)
		self.thetaTmp.fill_(0)
		self.thetaTmp += torch.tensor([[-50,-50], [100,-100], [-100,100]], dtype=torch.double)
		self.thetaTmp2 = self.nn.getLayer(1)
		self.thetaTmp2 += torch.tensor([[-0.2], [1], [1]], dtype=torch.double)
	def __call__(self, x, y):
		self.x = x
		self.y = y
		result = self.forward()
		if result > 0.5: return True
		else: return False
	def forward(self):
		return self.nn.forward(torch.tensor([self.x, self.y],dtype=torch.double))