# Model Building 

This problem can be formulated as two subproblems, (a) Image segmentation and (b) Depth estimation, where the given fg_bg and bg image of size H x W x C is used to predict a mask of size H x W x 1 and depth of H x W x 1

## Survey 

I started with the article on [Overview of Semantic Segmentation](https://www.jeremyjordan.me/semantic-segmentation/). To give a brief, the survey talks about the naive approach by stacking multiple convolutional layers with padding to preserve the size of the image. Later, it emphasizes on the encoder-decoder architecture. In such architectures, the image spatial resolution is downsampled with rich mappings, considered as encoded image. The decoder part upsamples the feature representation and produces the full-resolution output. The article led to another paper [Resunet-a](https://arxiv.org/abs/1904.00592). This is advanced version of the Unet architecture. It is an encoder-decoder architecture, where the residual convolutional layers are used. 
I performed a quick glance on another article on depth prediction to understand how the supervised learning of depth prediction is done. [Guide for depth estimation](https://heartbeat.fritz.ai/research-guide-for-depth-estimation-with-deep-learning-1a02a439b834) highlighted multiple approaches. The article let to the paper on [Deeper Depth Prediction with Fully Convolutional Residual Networks] (https://arxiv.org/abs/1606.00373v2) 
These architectures motivation the architecture I used to solve the given problem. 

## Design 

### Network architecture 

The architecture uses the ResUnet
[Unet Architecture] 

About the variants of the architecture - Lite mode. 

### Loss functions

This section talks about the analyis of different loss functions. 

#### BCEWithLogits

#### L1 Loss 

#### Dice Loss 

#### SSIM 

#### Root Mean Squared Error Loss 

#### HuberLoss

#### SpatialGradient

#### Berhu Loss

### Metrics

#### IOU

#### Pixel Accuracy 

## Implementation Detail

To start with, I created a tiny dataset out of the bigger set containing randomly picked up 20k fg_bg images with their corresponding mask and depth images. The tiny data set can be accessed at [tiny_data.zip](https://drive.google.com/open?id=1Tw2Ijf2l7fERsOEQ7k_IMRXYTzm4BaI4). 

### Optimizations 


### Bug Fixes


