
# coding: utf-8

# In[1]:


from neural_network import NeuralNetwork
from nn2 import NeuralNetwork2
import matplotlib.pyplot as plt
import torch
from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR


# In[2]:


And = AND()
#print(And.nn.theta)
print(And(True, True))
print(And(True, False))
print(And(False, True))
print(And(False, False))


# In[3]:


Or = OR()
#print(Or.nn.theta)
print(Or(True, True))
print(Or(True, False))
print(Or(False, True))
print(Or(False, False))


# In[4]:


Not = NOT()
#print(Not.nn.theta)
print(Not(False))
print(Not(True))


# In[5]:


Xor = XOR()
#print(Xor.nn.theta)
print(Xor(True, True))
print(Xor(True, False))
print(Xor(False, True))
print(Xor(False, False))


# In[6]:


nn = NeuralNetwork(2,2,1)
#nn2 = NeuralNetwork(2,3,1)


# In[7]:


print(nn.Theta)
#print(nn2.Theta)
#nn2.Theta['Theta(layer0-layer1)'] = nn.Theta[0].float()
#nn2.Theta['Theta(layer1-layer2)'] = (nn.Theta[1]).t().float()
print(nn.Theta)
#print(nn2.Theta)


# In[8]:


nn.forward(torch.tensor([0,1], dtype=torch.double))
#nn2.forward(torch.tensor([3,4], dtype=torch.double))


# In[9]:


print(nn.a)
print(nn.z)
#print(nn2.a)
#print(nn2.z)


# In[10]:


#nn.forward(torch.tensor([[21,35,87,90],[90,23,45,12]],dtype=torch.double))
#nn2.forward(torch.tensor([[21,35,87,90],[90,23,45,12]],dtype=torch.float))


# In[11]:


print(nn.a)
print(nn.z)
#print(nn2.a)
#print(nn2.z)


# In[12]:


nn.backward(torch.tensor([[1]], dtype=torch.double))
#nn2.backward(torch.tensor([[1,2,3,4]], dtype=torch.double))


# In[13]:


print(nn.dE_dTheta)
#print(nn2.dE_dTheta)


# In[14]:


print(nn.Theta)
nn.updateParams(0.01)
print(nn.Theta)


# In[15]:


And.train()
print(And(True, True))
print(And(True, False))
print(And(False, True))
print(And(False, False))


# In[16]:


Or.train()
print(Or(True, True))
print(Or(True, False))
print(Or(False, True))
print(Or(False, False))


# In[17]:


Not.train()
print(Not(False))
print(Not(True))


# In[18]:


Xor.train()
print(Xor(True, True))
print(Xor(True, False))
print(Xor(False, True))
print(Xor(False, False))

