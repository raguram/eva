## Model Building 

I am considering this as an image segmentation problem, where given the fg_bg and the bg image of size H x W x C, I want to predict a mask of size H x W x 1 and depth of H x W x 1. 

### Survey 

[Talk about different papers]
During the survey of existing solutions for image segmentations, I came across https://www.jeremyjordan.me/semantic-segmentation. 

### Design 

##### Network architecture 

The architecture uses the ResUnet
[Unet Architecture] 

About the variants of the architecture - Lite mode. 

##### Loss functions

This section talks about the analyis of different loss functions. 

BCEWithLogits

L1 Loss 

Dice Loss 

SSIM 

Root Mean Squared Error Loss 

HuberLoss

SpatialGradient

Berhu Loss

###### Metrics

IOU

Pixel Accuracy 

### Implementation Detail

To start with, I created a tiny dataset out of the bigger set containing randomly picked up 20k fg_bg images with their corresponding mask and depth images. The tiny data set can be accessed at [tiny_data.zip](https://drive.google.com/open?id=1Tw2Ijf2l7fERsOEQ7k_IMRXYTzm4BaI4). 

##### Optimizations 


##### Bug Fixes


