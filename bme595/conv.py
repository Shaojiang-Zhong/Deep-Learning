import torch
import numpy as np
class Conv2D:
    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        self.in_channel = in_channel
        self.o_channel = o_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode
        self.K1 = [[-1, -1, -1],
                   [0, 0, 0],
                   [1, 1, 1]]
        self.K2 = [[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]]
        self.K3 = [[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]
        self.K4 = [[-1, -1, -1, -1, -1],
                   [-1, -1, -1, -1, -1],
                   [0, 0, 0, 0, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1]]
        self.K5 = [[-1, -1, 0, 1, 1],
                   [-1, -1, 0, 1, 1],
                   [-1, -1, 0, 1, 1],
                   [-1, -1, 0, 1, 1],
                   [-1, -1, 0, 1, 1]]

        pass
    def addPadding(self, input_image, kernel_size):
        padSize = kernel_size//2
        imageList = input_image.tolist()
        for row in imageList:
            row = padSize*[0] + row + padSize*[0]
        for i in range(padSize):
            imageList = [len(imageList[0])*[0]] + imageList
        for i in range(padSize):
            imageList = imageList + [len(imageList[0])*[0]] 

        #np.savetxt('testimage1.txt', (np.asarray(imageList)).ravel(), fmt="%f")

        print("here!!")


        return torch.Tensor(imageList)
    def colorToGrey(self, input_image):
        return (input_image[0] + input_image[1] + input_image[2])/3
    def getKernelArr(self, kernelArr):
        #padding is 
        
        if self.mode == 'known':
            if self.o_channel == 1:
                #use K1
                kernelArr.append(self.K1)
                pass
            elif self.o_channel == 2:
                #use K4, K5
                kernelArr.append(self.K4)
                kernelArr.append(self.K5)
                pass
            else:
                kernelArr.append(self.K1)
                kernelArr.append(self.K2)
                kernelArr.append(self.K3)
                pass
        else:
            #generate random kernal
            for i in range(self.o_channel):
                kernelArr.append(torch.randn(self.kernel_size, self.kernel_size))
            pass
    def getTargetList(self, greyImage):

        #return torch.zeros(self.o_channel, m, n)
        retList = []
        for i in range(self.o_channel):
          retList.append(np.copy(greyImage.numpy()))
        retList = np.asarray(retList)
        return torch.from_numpy(retList)
    def forward(self, input_image):
        #check if input_image has 3 in_channel, return False if not
        kernelArr = []
        #get kernelArr
        self.getKernelArr(kernelArr)
        #checkIfcolored(input_image) double check
        greyImage = self.colorToGrey(input_image)
        # padding image
        padImage = self.addPadding(greyImage, self.kernel_size)


        targetList = self.getTargetList(greyImage)

        operation_count = 0
        # loop and calculate
        for i in range(self.kernel_size//2, len(padImage)-self.kernel_size//2, self.stride):
            for j in range(self.kernel_size//2, len(padImage[0])-self.kernel_size//2, self.stride):
                # (i,j) pixel in the image


                for q in range(len(kernelArr)):
                    # for this specific kernel
                    kernel = kernelArr[q]
                    tmpSum = 0


                    for k in range(0, self.kernel_size):
                        for p in range(0, self.kernel_size):
                            tmpSum += kernel[k][p] * padImage[i-self.kernel_size//2+k][j-self.kernel_size//2+p]
                            operation_count += 2
                    operation_count -= 1

                    targetList[q][i-self.kernel_size//2][j-self.kernel_size//2] = tmpSum

        return [operation_count, targetList]
