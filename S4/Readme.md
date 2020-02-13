## Problem Statement 

The goal of this assignment is to train a MNIST model having validation accuracy of atleast 99.4% with the following constraints. 

* 99.4% validation accuracy
* Less than 20k Parameters
* Less than 20 Epochs
* No fully connected layer

## Solution 

In this part, I will describe each of the iterations to achieve the above goal. 

#### Iteration 0

I started with the same architecture as provided in our second assignment. The architecture has 6MM parameters. This is to ensure that code I have written works correctly. 
Link - https://github.com/raguram/eva/commit/22275217ebc773da1508655389d7cdd9c36ed182#diff-c91008d9ecc1755dff6badd8d231ed7c

#### Iteration 1 - Accuracy - 98.42%

https://github.com/raguram/eva/blob/9b3348df7e41504a2760dd2e9888b50d78c26723/S4/MNIST.ipynb

* Architecture: I used a simple model with 14,410 parameters using only Conv (3x3), Relu, Max pool, GAP and Soft Max. 
* Data: No data transform

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------        
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
            Conv2d-3           [-1, 16, 28, 28]           2,320
              ReLU-4           [-1, 16, 28, 28]               0
         MaxPool2d-5           [-1, 16, 14, 14]               0
            Conv2d-6           [-1, 16, 14, 14]           2,320
              ReLU-7           [-1, 16, 14, 14]               0
            Conv2d-8           [-1, 16, 14, 14]           2,320
              ReLU-9           [-1, 16, 14, 14]               0
        MaxPool2d-10             [-1, 16, 7, 7]               0
           Conv2d-11             [-1, 16, 7, 7]           2,320
             ReLU-12             [-1, 16, 7, 7]               0
           Conv2d-13             [-1, 32, 7, 7]           4,640
             ReLU-14             [-1, 32, 7, 7]               0
           Conv2d-15             [-1, 10, 7, 7]             330
        AdaptiveAvgPool2d-16     [-1, 10, 1, 1]               0
----------------------------------------------------------------
* Total params: 14,410
* Trainable params: 14,410

#### Iteration 2 - 99.35%

https://github.com/raguram/eva/blob/2728f2572eae53684938d1a6abebbab4ea6f8eb2/S4/MNIST.ipynb
* Architecture: 20,394 parameters using only Conv (3x3), Relu, BatchNorm, Dropout (0.05), Conv (1x1), Max pool, GAP and Soft Max. 
* Data: Normalized data with Mean and Variance. 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
         Dropout2d-4           [-1, 16, 28, 28]               0
            Conv2d-5           [-1, 32, 28, 28]           4,640
              ReLU-6           [-1, 32, 28, 28]               0
       BatchNorm2d-7           [-1, 32, 28, 28]              64
         Dropout2d-8           [-1, 32, 28, 28]               0
            Conv2d-9           [-1, 16, 28, 28]             528
        MaxPool2d-10           [-1, 16, 14, 14]               0
           Conv2d-11           [-1, 16, 14, 14]           2,320
             ReLU-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
        Dropout2d-14           [-1, 16, 14, 14]               0
           Conv2d-15           [-1, 32, 14, 14]           4,640
             ReLU-16           [-1, 32, 14, 14]               0
      BatchNorm2d-17           [-1, 32, 14, 14]              64
        Dropout2d-18           [-1, 32, 14, 14]               0
           Conv2d-19           [-1, 16, 14, 14]             528
             ReLU-20           [-1, 16, 14, 14]               0
        MaxPool2d-21             [-1, 16, 7, 7]               0
           Conv2d-22             [-1, 16, 7, 7]           2,320
             ReLU-23             [-1, 16, 7, 7]               0
      BatchNorm2d-24             [-1, 16, 7, 7]              32
        Dropout2d-25             [-1, 16, 7, 7]               0
           Conv2d-26             [-1, 32, 7, 7]           4,640
             ReLU-27             [-1, 32, 7, 7]               0
      BatchNorm2d-28             [-1, 32, 7, 7]              64
        Dropout2d-29             [-1, 32, 7, 7]               0
           Conv2d-30             [-1, 10, 7, 7]             330
    AdaptiveAvgPool2d-31         [-1, 10, 1, 1]               0
----------------------------------------------------------------

* Total params: 20,394
* Trainable params: 20,394

#### Iteration 3 - 99.5%

https://github.com/raguram/eva/blob/1e883b59add81f0f907200b3b9cb35379297dc02/S4/MNIST.ipynb
In this iteration, I was majorly playing around the dropout, doing a binary search to hit the optimal value. Architecture: 20,394 parameters using only Conv (3x3), Relu, BatchNorm, Dropout (0.075), Conv (1x1), Max pool, GAP and Soft Max. 
Data: Normalized data with Mean and Variance. 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
         Dropout2d-4           [-1, 16, 28, 28]               0
            Conv2d-5           [-1, 32, 28, 28]           4,640
              ReLU-6           [-1, 32, 28, 28]               0
       BatchNorm2d-7           [-1, 32, 28, 28]              64
         Dropout2d-8           [-1, 32, 28, 28]               0
            Conv2d-9           [-1, 16, 28, 28]             528
        MaxPool2d-10           [-1, 16, 14, 14]               0
           Conv2d-11           [-1, 16, 14, 14]           2,320
             ReLU-12           [-1, 16, 14, 14]               0
      BatchNorm2d-13           [-1, 16, 14, 14]              32
        Dropout2d-14           [-1, 16, 14, 14]               0
           Conv2d-15           [-1, 32, 14, 14]           4,640
             ReLU-16           [-1, 32, 14, 14]               0
      BatchNorm2d-17           [-1, 32, 14, 14]              64
        Dropout2d-18           [-1, 32, 14, 14]               0
           Conv2d-19           [-1, 16, 14, 14]             528
             ReLU-20           [-1, 16, 14, 14]               0
        MaxPool2d-21             [-1, 16, 7, 7]               0
           Conv2d-22             [-1, 16, 7, 7]           2,320
             ReLU-23             [-1, 16, 7, 7]               0
      BatchNorm2d-24             [-1, 16, 7, 7]              32
        Dropout2d-25             [-1, 16, 7, 7]               0
           Conv2d-26             [-1, 32, 7, 7]           4,640
             ReLU-27             [-1, 32, 7, 7]               0
      BatchNorm2d-28             [-1, 32, 7, 7]              64
        Dropout2d-29             [-1, 32, 7, 7]               0
           Conv2d-30             [-1, 10, 7, 7]             330
    AdaptiveAvgPool2d-31         [-1, 10, 1, 1]               0
----------------------------------------------------------------
* Total params: 20,394
* Trainable params: 20,394

#### Iteration 4 - 99.39%

https://github.com/raguram/eva/blob/87b812e416258e875b3bde1198e2cb40624aa011/S4/MNIST.ipynb
In this iteration, I was trying to reduce the number of parameters to be lesser than 20k. Looking at the misclassifications, one of the things I noticed was that there were lesser number of layers in the first filter group (RF was only 5x5). I added one more layer in the first filter group to increase the RF to 7x7 before applying Max pool. I played with different kernel sizes. 
* Architecture: 20,394 parameters using only Conv (3x3), Relu, BatchNorm, Dropout (0.075), Conv (1x1), Max pool, GAP and Soft Max. 
* Data: Normalized data with Mean and Variance. 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
         Dropout2d-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 16, 28, 28]           1,168
              ReLU-6           [-1, 16, 28, 28]               0
       BatchNorm2d-7           [-1, 16, 28, 28]              32
         Dropout2d-8           [-1, 16, 28, 28]               0
            Conv2d-9           [-1, 32, 28, 28]           4,640
             ReLU-10           [-1, 32, 28, 28]               0
      BatchNorm2d-11           [-1, 32, 28, 28]              64
        Dropout2d-12           [-1, 32, 28, 28]               0
           Conv2d-13            [-1, 8, 28, 28]             264
        MaxPool2d-14            [-1, 8, 14, 14]               0
           Conv2d-15           [-1, 16, 14, 14]           1,168
             ReLU-16           [-1, 16, 14, 14]               0
      BatchNorm2d-17           [-1, 16, 14, 14]              32
        Dropout2d-18           [-1, 16, 14, 14]               0
           Conv2d-19           [-1, 32, 14, 14]           4,640
             ReLU-20           [-1, 32, 14, 14]               0
      BatchNorm2d-21           [-1, 32, 14, 14]              64
        Dropout2d-22           [-1, 32, 14, 14]               0
           Conv2d-23            [-1, 8, 14, 14]             264
             ReLU-24            [-1, 8, 14, 14]               0
        MaxPool2d-25              [-1, 8, 7, 7]               0
           Conv2d-26             [-1, 16, 7, 7]           1,168
             ReLU-27             [-1, 16, 7, 7]               0
      BatchNorm2d-28             [-1, 16, 7, 7]              32
        Dropout2d-29             [-1, 16, 7, 7]               0
           Conv2d-30             [-1, 32, 7, 7]           4,640
             ReLU-31             [-1, 32, 7, 7]               0
      BatchNorm2d-32             [-1, 32, 7, 7]              64
        Dropout2d-33             [-1, 32, 7, 7]               0
           Conv2d-34             [-1, 10, 7, 7]             330
    AdaptiveAvgPool2d-35         [-1, 10, 1, 1]               0
----------------------------------------------------------------
* Total params: 18,666
* Trainable params: 18,666

#### Iteration 5 - 99.4%

https://github.com/raguram/eva/blob/d36bda90bc72f6aa708d815ec63a9ae701ec9728/S4/MNIST.ipynb
Same architecture as above. Added data transformations. Reason: Misclassifications were coming from slight rotation of the digits or missing pixels that differentiats 7 from a 9. So, tried with data transformation. Or may be its just that I got lucky this time :-D 
Also, I noticed with 20 epochs the model was still converging (improving). When I trained the previous model for higher epochs (40), I was getting an accuracy of 99.47%. So, I thought increasing the learning rate would help. So, changed lr to 0.02. 
* Architecture: 20,394 parameters using only Conv (3x3), Relu, BatchNorm, Dropout (0.075), Conv (1x1), Max pool, GAP and Soft Max. 
* Data: Normalized data with Mean and Variance. RandomAffine and ColorJitter (ref: https://www.kaggle.com/enwei26/mnist-digits-pytorch-cnn-99)  

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1            [-1, 8, 28, 28]              80
              ReLU-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
         Dropout2d-4            [-1, 8, 28, 28]               0
            Conv2d-5           [-1, 16, 28, 28]           1,168
              ReLU-6           [-1, 16, 28, 28]               0
       BatchNorm2d-7           [-1, 16, 28, 28]              32
         Dropout2d-8           [-1, 16, 28, 28]               0
            Conv2d-9           [-1, 32, 28, 28]           4,640
             ReLU-10           [-1, 32, 28, 28]               0
      BatchNorm2d-11           [-1, 32, 28, 28]              64
        Dropout2d-12           [-1, 32, 28, 28]               0
           Conv2d-13            [-1, 8, 28, 28]             264
        MaxPool2d-14            [-1, 8, 14, 14]               0
           Conv2d-15           [-1, 16, 14, 14]           1,168
             ReLU-16           [-1, 16, 14, 14]               0
      BatchNorm2d-17           [-1, 16, 14, 14]              32
        Dropout2d-18           [-1, 16, 14, 14]               0
           Conv2d-19           [-1, 32, 14, 14]           4,640
             ReLU-20           [-1, 32, 14, 14]               0
      BatchNorm2d-21           [-1, 32, 14, 14]              64
        Dropout2d-22           [-1, 32, 14, 14]               0
           Conv2d-23            [-1, 8, 14, 14]             264
             ReLU-24            [-1, 8, 14, 14]               0
        MaxPool2d-25              [-1, 8, 7, 7]               0
           Conv2d-26             [-1, 16, 7, 7]           1,168
             ReLU-27             [-1, 16, 7, 7]               0
      BatchNorm2d-28             [-1, 16, 7, 7]              32
        Dropout2d-29             [-1, 16, 7, 7]               0
           Conv2d-30             [-1, 32, 7, 7]           4,640
             ReLU-31             [-1, 32, 7, 7]               0
      BatchNorm2d-32             [-1, 32, 7, 7]              64
        Dropout2d-33             [-1, 32, 7, 7]               0
           Conv2d-34             [-1, 10, 7, 7]             330
    AdaptiveAvgPool2d-35         [-1, 10, 1, 1]               0
----------------------------------------------------------------
* Total params: 18,666
* Trainable params: 18,666
