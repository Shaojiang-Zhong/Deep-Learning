
# coding: utf-8

# In[10]:


from conv import Conv2D
import torch 
from PIL import Image
from torchvision.transforms import ToTensor
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time


# In[11]:


image1 = Image.open('testImage1.jpg')
image1 = ToTensor()(image1).unsqueeze(0)
image1=Variable(image1)
image1 = image1[0]


# In[12]:


image2 = Image.open('testImage2.jpeg')
image2 = ToTensor()(image2).unsqueeze(0)
image2=Variable(image2)
image2 = image2[0]


# In[13]:


def saveImage(imageTensor, imageNum, taskNum, kernalNum):
    output_image = imageTensor.numpy()
    output_img_norm=(((output_image[kernalNum,:,:] - output_image[kernalNum,:,:].min()) / output_image[kernalNum,:,:].max()-output_image[kernalNum,:,:].min()) * 255.0).astype(np.uint8)
    output_img_gray = Image.fromarray(output_img_norm)
    image_name = 'out_%s_task_%s_%s.jpg'%(imageNum,taskNum,kernalNum)
    output_img_gray.save(image_name)


# In[14]:


"""
Part A.
Initialize Conv2D in main.py (conv2d = Conv2D(*args)) for one of the task.
Call conv2d.forward() with your input image [3D FloatTensor]. The forward() function must return output [int, 3D FloatTensor].
Save each channel of output tensor separately as a grayscale image in your main repository.
Repeat 2-4 for all the three tasks.
"""


# In[15]:


#task 1
start = time.time()
conv2d = Conv2D(in_channel=3, o_channel=1, kernel_size=3, stride=1, mode='known')
Number_of_ops, output_image = conv2d.forward(image1)
end = time.time()
print(Number_of_ops)
print(start-end)
print(output_image.size())
saveImage(output_image, 1, 1, 0)


# In[ ]:


#task 2
start = time.time()
conv2d = Conv2D(in_channel=3, o_channel=2, kernel_size=5, stride=1, mode='known')
Number_of_ops, output_image = conv2d.forward(image1)
end = time.time()
print(start-end)
print(Number_of_ops)
print(output_image.size())
for i in range(2):
    saveImage(imageTensor=output_image, imageNum=1, taskNum=2, kernalNum=i)


# In[ ]:


#task 3
start = time.time()
conv2d = Conv2D(in_channel=3, o_channel=3, kernel_size=3, stride=2, mode='known')
Number_of_ops, output_image = conv2d.forward(image1)
end = time.time()
print(start-end)
print(Number_of_ops)
print(output_image.size())
for i in range(3):
    saveImage(imageTensor=output_image, imageNum=1, taskNum=3, kernalNum=i)


# In[15]:


'''
Part B
Initialize Conv2D using values of Task 1 and set o_channel to 2^i (i = 0, 1, …, 10) and mode=’rand’.
Plot the time taken for performing each forward() pass as a function of i.
'''


# In[ ]:


import time
import matplotlib.pyplot as plt

timeList = []
iList = []
for i in range(0,11):
    start = time.time()
    conv2d = Conv2D(in_channel=3, o_channel=2**i, kernel_size=3, stride=1, mode='rand')
    Number_of_ops, output_image = conv2d.forward(image1)
    end = time.time()
    print(i)
    print(Number_of_ops)
    print(output_image.size())
    print(end-start)
    timeList.append(end-start)
    iList.append(i)


# In[ ]:


import time
import matplotlib.pyplot as plt

timeList = []
iList = []
i = 5
start = time.time()
conv2d = Conv2D(in_channel=3, o_channel=2**i, kernel_size=3, stride=1, mode='rand')
Number_of_ops, output_image = conv2d.forward(image1)
end = time.time()
print(i)
print(Number_of_ops)
print(output_image.size())
print(end-start)
timeList.append(end-start)
iList.append(i)


# In[ ]:


plt.plot(iList, timeList, 'ro')
plt.axis([0, 12, 0, timeList[-1]])
plt.show()


# In[ ]:


"""
Part C
Initialize Conv2D using values of Task 2 with kernel_size=3, 5, …, 11 and mode=’rand’.
Plot number of operations (int returned by forward()) used to perform convolution as a function of kernel_size.
"""


# In[10]:


operationList = []
ksList = []
for ks in range(3,13,2):
    conv2d = Conv2D(in_channel=3, o_channel=2, kernel_size=ks, stride=1, mode='rand')
    Number_of_ops, output_image = conv2d.forward(image1)
    print(ks)
    print(Number_of_ops)
    print(output_image.size())
    operationList.append(Number_of_ops)
    ksList.append(ks)
plt.plot(ksList, operationList, 'ro')
plt.axis([0, 13, 0, operationList[-1]])
plt.show()

