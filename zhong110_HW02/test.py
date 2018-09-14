
# coding: utf-8

# In[6]:


from neural_network import NeuralNetwork
import torch
from logic_gates import AND
from logic_gates import OR
from logic_gates import NOT
from logic_gates import XOR


# In[7]:


And = AND()
#print(And.nn.theta)
print(And(True, True))
print(And(True, False))
print(And(False, True))
print(And(False, False))


# In[8]:


Or = OR()
#print(Or.nn.theta)
print(Or(True, True))
print(Or(True, False))
print(Or(False, True))
print(Or(False, False))


# In[9]:


Not = NOT()
#print(Not.nn.theta)
print(Not(False))
print(Not(True))


# In[10]:


Xor = XOR()
#print(Xor.nn.theta)
print(Xor(True, True))
print(Xor(True, False))
print(Xor(False, True))
print(Xor(False, False))

