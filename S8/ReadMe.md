## Problem Statement 

- Extract the ResNet18 model from this repository and add it to your API/repo. 
- Use your data loader, model loading, train, and test code to train ResNet18 on Cifar10
- Your Target is 85% accuracy. No limit on the number of epochs. Use default ResNet18 code (so params are fixed). 

## Solution

- Modularized library is used for training a model for CIFAR10 with Resnet18 - https://github.com/raguram/eva/tree/master/cnn-lib
- Max test accuracy = 85.93% 
- Number of epochs = 20

### Comments 

Though the test accuracy has met the target 85%, the model is overfitting. TODO: Add appropriate regularization.
