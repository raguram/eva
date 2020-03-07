## Problem Statement 

Train a model for CIFAR10 with following constraints

* total RF must be more than 44
* one of the layers must use Depthwise Separable Convolution
* one of the layers must use Dilated Convolution
* use GAP (compulsory):- add FC after GAP to target #of classes (optional)
* achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M. 

## Solution 

* Total RF: 50 
* Depthwise Seperable convolution: Used 
* Dilated Convolution: Used 
* GAP: Used 
* Validation Accuracy: 82.73 (37th Epoch) 
* Number of epochs: 40 
* Number of parameters: 223,520

Link: https://github.com/raguram/eva/blob/master/S7/CIFAR10_Assignment7_withAdvancedConv.ipynb

## Model Summary 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 32, 30, 30]             864
              ReLU-2           [-1, 32, 30, 30]               0
       BatchNorm2d-3           [-1, 32, 30, 30]              64
            Conv2d-4           [-1, 32, 28, 28]           9,216
              ReLU-5           [-1, 32, 28, 28]               0
       BatchNorm2d-6           [-1, 32, 28, 28]              64
         Dropout2d-7           [-1, 32, 28, 28]               0
         MaxPool2d-8           [-1, 32, 14, 14]               0
            Conv2d-9           [-1, 64, 12, 12]          18,432
             ReLU-10           [-1, 64, 12, 12]               0
      BatchNorm2d-11           [-1, 64, 12, 12]             128
        Dropout2d-12           [-1, 64, 12, 12]               0
           Conv2d-13           [-1, 64, 10, 10]          36,864
             ReLU-14           [-1, 64, 10, 10]               0
      BatchNorm2d-15           [-1, 64, 10, 10]             128
        Dropout2d-16           [-1, 64, 10, 10]               0
        MaxPool2d-17             [-1, 64, 5, 5]               0
           Conv2d-18             [-1, 64, 5, 5]             576
           Conv2d-19            [-1, 128, 5, 5]           8,192
             ReLU-20            [-1, 128, 5, 5]               0
      BatchNorm2d-21            [-1, 128, 5, 5]             256
        Dropout2d-22            [-1, 128, 5, 5]               0
           Conv2d-23            [-1, 128, 5, 5]         147,456
    AdaptiveAvgPool2d-24        [-1, 128, 1, 1]               0
           Conv2d-25             [-1, 10, 1, 1]           1,280


* Total params: 223,520
* Trainable params: 223,520

