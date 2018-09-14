from neural_network import NeuralNetwork
import torch
import matplotlib.pyplot as plt
class AND:
	def __init__(self):
		self.nn = NeuralNetwork(2,1)
		self.maxLoop = 10000
		#print(self.nn.theta)
		#self.thetaTmp = self.nn.getLayer(0)
		#self.thetaTmp.fill_(0)
		#self.thetaTmp += torch.tensor([[-1.5], [1], [1]], dtype=torch.double)
	def __call__(self, x, y):
		self.x = x
		self.y = y
		result = self.forward()
		if result > 0.5: return True
		else: return False
	def forward(self):
		return self.nn.forward(torch.tensor([self.x, self.y],dtype=torch.double))
	def train(self):
		inputSet = torch.tensor([[0,0,1,1],[0,1,0,1]], dtype=torch.double)
		targetSet = torch.tensor([0,0,0,1], dtype=torch.double)
		for i in range(self.maxLoop):
			self.nn.forward(inputSet)
			self.nn.backward(targetSet)
			plt.plot(i, self.nn.lossValue, '.r-')
			plt.xlabel('iteration number')
			plt.ylabel('Mean Square Error')
			plt.title('iteration number vs Mean Square Error')
			plt.grid(True)
			if self.nn.lossValue > 0.01:
				self.nn.updateParams(1.0)
			else:
				#bingo
				print(self.nn.Theta)
				break
		plt.show()
		print("loop ends, here's the theta")
		print(self.nn.Theta)
		pass
class OR:
	def __init__(self):
		self.nn = NeuralNetwork(2,1)
		self.maxLoop = 10000
		#print(self.nn.theta)
		#self.thetaTmp = self.nn.getLayer(0)
		#self.thetaTmp.fill_(0)
		#self.thetaTmp += torch.tensor([[-0.5], [1], [1]], dtype=torch.double)
	def __call__(self, x, y):
		self.x = x
		self.y = y
		result = self.forward()
		if result > 0.5: return True
		else: return False
	def forward(self):
		return self.nn.forward(torch.tensor([self.x, self.y],dtype=torch.double))
	def train(self):
		inputSet = torch.tensor([[0,0,1,1],[0,1,0,1]], dtype=torch.double)
		targetSet = torch.tensor([0,1,1,1], dtype=torch.double)
		for i in range(self.maxLoop):
			self.nn.forward(inputSet)
			self.nn.backward(targetSet)
			plt.plot(i, self.nn.lossValue, '.r-')
			plt.xlabel('iteration number')
			plt.ylabel('Mean Square Error')
			plt.title('iteration number vs Mean Square Error')
			plt.grid(True)
			if self.nn.lossValue > 0.01:
				self.nn.updateParams(1.0)
			else:
				#bingo
				print(self.nn.Theta)
				break
		plt.show()
		print("loop ends, here's the theta")
		print(self.nn.Theta)
		pass
class NOT:
	def __init__(self):
		self.nn = NeuralNetwork(1,1)
		self.maxLoop = 10000
		#print(self.nn.theta)
		#self.thetaTmp = self.nn.getLayer(0)
		#self.thetaTmp.fill_(0)
		#self.thetaTmp += torch.tensor([[0.5], [-1]], dtype=torch.double)
	def __call__(self, x):
		self.x = x
		result = self.forward()
		if result > 0.5: return True
		else: return False
	def forward(self):
		return self.nn.forward(torch.tensor([self.x],dtype=torch.double))
	def train(self):
		inputSet = torch.tensor([[0,1]], dtype=torch.double)
		targetSet = torch.tensor([1,0], dtype=torch.double)
		for i in range(self.maxLoop):
			self.nn.forward(inputSet)
			self.nn.backward(targetSet)
			plt.plot(i, self.nn.lossValue, '.r-')
			plt.xlabel('iteration number')
			plt.ylabel('Mean Square Error')
			plt.title('iteration number vs Mean Square Error')
			plt.grid(True)
			if self.nn.lossValue > 0.01:
				self.nn.updateParams(1.0)
			else:
				#bingo
				print(self.nn.Theta)
				break
		plt.show()
		print("loop ends, here's the theta")
		print(self.nn.Theta)
		pass

class XOR:
	def __init__(self):
		self.nn = NeuralNetwork(2,2,1)
		self.maxLoop = 10000
		#print(self.nn.theta)
		#self.thetaTmp = self.nn.getLayer(0)
		#self.thetaTmp.fill_(0)
		#self.thetaTmp += torch.tensor([[-50, -50], [60, -60], [-60, 60]], dtype=torch.double)
		#self.thetaTmp2 = self.nn.getLayer(1)
		#self.thetaTmp2 += torch.tensor([[-50], [60], [60]], dtype=torch.double)
	def __call__(self, x, y):
		self.x = x
		self.y = y
		result = self.forward()
		if result > 0.5: return True
		else: return False
	def forward(self):
		return self.nn.forward(torch.tensor([self.x, self.y],dtype=torch.double))
	def train(self):
		inputSet = torch.tensor([[0,0,1,1],[0,1,0,1]], dtype=torch.double)
		targetSet = torch.tensor([0,1,1,0], dtype=torch.double)
		for i in range(self.maxLoop):
			self.nn.forward(inputSet)
			self.nn.backward(targetSet)
			#plt.plot(i, self.nn.lossValue, '.r-')
			#plt.xlabel('iteration number')
			#plt.ylabel('Mean Square Error')
			#plt.title('iteration number vs Mean Square Error')
			#plt.grid(True)
			if self.nn.lossValue > 0.01:
				self.nn.updateParams(1.0)
			else:
				#bingo
				print(self.nn.Theta)
				break
		#plt.show()
		print("loop ends, here's the theta")
		print(self.nn.Theta)
		pass