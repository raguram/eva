# Model Building 

This problem can be formulated as two subproblems, (a) Image segmentation and (b) Depth estimation, where the given fg_bg and bg image of size H x W x 3 is used to predict a mask of size H x W x 1 and depth of H x W x 1

## Survey 

I started with the article on [Overview of Semantic Segmentation](https://www.jeremyjordan.me/semantic-segmentation/). To give a brief, the survey talks about the naive approach by stacking multiple convolutional layers with padding to preserve the size of the image. Later, it emphasizes on the encoder-decoder architecture. In such architectures, the image spatial resolution is downsampled with rich mappings, considered as encoded image. The decoder part upsamples the feature representation and produces the full-resolution output. The article led to another paper [Resunet-a](https://arxiv.org/abs/1904.00592). This is advanced version of the Unet architecture. It is an encoder-decoder architecture, where the residual convolutional layers are used. 

I performed a quick glance on another article on depth prediction to understand how the supervised learning of depth prediction is done. [Guide for depth estimation](https://heartbeat.fritz.ai/research-guide-for-depth-estimation-with-deep-learning-1a02a439b834) highlighted multiple approaches. The article let to the paper on [Deeper Depth Prediction with Fully Convolutional Residual Networks] (https://arxiv.org/abs/1606.00373v2) 
These architectures motivation the architecture I used to solve the given problem. 

## Design 

### Network architecture 

Network architecture is an encoder-decoder architecture motivated from Resunet. Key components of the architecture are: 

- Encoder uses residual blocks with max pooling for downsampling. It has 4 residual blocks stacked up, getting the size of the original image by 16 times. The encoder gets the fg_bg and bg images concatinated. Hence, the input is of the size H x W x 6. 
- Double conv layer separating the encoder and decoder. 
- Mask decoder uses upsampling and convolutional layers to generate mask predictions. 
- Similary depth decoder uses upsampling and convolutional layers to generate depth predictions. 
[TODO: Add the architecture diagram]

Summary of the architecture can be found in the appendix below. You can access the code [here](https://github.com/raguram/eva/blob/master/cnn-lib/src/cnnlib/models/ResUNet.py#L87)

- Total params: 11,033,922
- Trainable params: 11,033,922
- Non-trainable params: 0

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

## Appendix 

#### Summary of the architecture 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 32, 96, 96]             192
       BatchNorm2d-2           [-1, 32, 96, 96]              64
            Conv2d-3           [-1, 32, 96, 96]           1,760
       BatchNorm2d-4           [-1, 32, 96, 96]              64
              ReLU-5           [-1, 32, 96, 96]               0
            Conv2d-6           [-1, 32, 96, 96]           9,248
       BatchNorm2d-7           [-1, 32, 96, 96]              64
              ReLU-8           [-1, 32, 96, 96]               0
        DoubleConv-9           [-1, 32, 96, 96]               0
             ReLU-10           [-1, 32, 96, 96]               0
        MaxPool2d-11           [-1, 32, 48, 48]               0
         ResBlock-12  [[-1, 32, 48, 48], [-1, 32, 96, 96]]               0
           Conv2d-13           [-1, 64, 48, 48]           2,048
      BatchNorm2d-14           [-1, 64, 48, 48]             128
           Conv2d-15           [-1, 64, 48, 48]          18,496
      BatchNorm2d-16           [-1, 64, 48, 48]             128
             ReLU-17           [-1, 64, 48, 48]               0
           Conv2d-18           [-1, 64, 48, 48]          36,928
      BatchNorm2d-19           [-1, 64, 48, 48]             128
             ReLU-20           [-1, 64, 48, 48]               0
       DoubleConv-21           [-1, 64, 48, 48]               0
             ReLU-22           [-1, 64, 48, 48]               0
        MaxPool2d-23           [-1, 64, 24, 24]               0
         ResBlock-24  [[-1, 64, 24, 24], [-1, 64, 48, 48]]               0
           Conv2d-25          [-1, 128, 24, 24]           8,192
      BatchNorm2d-26          [-1, 128, 24, 24]             256
           Conv2d-27          [-1, 128, 24, 24]          73,856
      BatchNorm2d-28          [-1, 128, 24, 24]             256
             ReLU-29          [-1, 128, 24, 24]               0
           Conv2d-30          [-1, 128, 24, 24]         147,584
      BatchNorm2d-31          [-1, 128, 24, 24]             256
             ReLU-32          [-1, 128, 24, 24]               0
       DoubleConv-33          [-1, 128, 24, 24]               0
             ReLU-34          [-1, 128, 24, 24]               0
        MaxPool2d-35          [-1, 128, 12, 12]               0
         ResBlock-36  [[-1, 128, 12, 12], [-1, 128, 24, 24]]               0
           Conv2d-37          [-1, 256, 12, 12]          32,768
      BatchNorm2d-38          [-1, 256, 12, 12]             512
           Conv2d-39          [-1, 256, 12, 12]         295,168
      BatchNorm2d-40          [-1, 256, 12, 12]             512
             ReLU-41          [-1, 256, 12, 12]               0
           Conv2d-42          [-1, 256, 12, 12]         590,080
      BatchNorm2d-43          [-1, 256, 12, 12]             512
             ReLU-44          [-1, 256, 12, 12]               0
       DoubleConv-45          [-1, 256, 12, 12]               0
             ReLU-46          [-1, 256, 12, 12]               0
        MaxPool2d-47            [-1, 256, 6, 6]               0
         ResBlock-48  [[-1, 256, 6, 6], [-1, 256, 12, 12]]               0
           Conv2d-49            [-1, 512, 6, 6]       1,180,160
      BatchNorm2d-50            [-1, 512, 6, 6]           1,024
             ReLU-51            [-1, 512, 6, 6]               0
           Conv2d-52            [-1, 512, 6, 6]       2,359,808
      BatchNorm2d-53            [-1, 512, 6, 6]           1,024
             ReLU-54            [-1, 512, 6, 6]               0
       DoubleConv-55            [-1, 512, 6, 6]               0
         Upsample-56          [-1, 512, 12, 12]               0
           Conv2d-57          [-1, 256, 12, 12]       1,769,728
      BatchNorm2d-58          [-1, 256, 12, 12]             512
             ReLU-59          [-1, 256, 12, 12]               0
           Conv2d-60          [-1, 256, 12, 12]         590,080
      BatchNorm2d-61          [-1, 256, 12, 12]             512
             ReLU-62          [-1, 256, 12, 12]               0
       DoubleConv-63          [-1, 256, 12, 12]               0
          UpBlock-64          [-1, 256, 12, 12]               0
         Upsample-65          [-1, 256, 24, 24]               0
           Conv2d-66          [-1, 128, 24, 24]         442,496
      BatchNorm2d-67          [-1, 128, 24, 24]             256
             ReLU-68          [-1, 128, 24, 24]               0
           Conv2d-69          [-1, 128, 24, 24]         147,584
      BatchNorm2d-70          [-1, 128, 24, 24]             256
             ReLU-71          [-1, 128, 24, 24]               0
       DoubleConv-72          [-1, 128, 24, 24]               0
          UpBlock-73          [-1, 128, 24, 24]               0
         Upsample-74          [-1, 128, 48, 48]               0
           Conv2d-75           [-1, 64, 48, 48]         110,656
      BatchNorm2d-76           [-1, 64, 48, 48]             128
             ReLU-77           [-1, 64, 48, 48]               0
           Conv2d-78           [-1, 64, 48, 48]          36,928
      BatchNorm2d-79           [-1, 64, 48, 48]             128
             ReLU-80           [-1, 64, 48, 48]               0
       DoubleConv-81           [-1, 64, 48, 48]               0
          UpBlock-82           [-1, 64, 48, 48]               0
         Upsample-83           [-1, 64, 96, 96]               0
           Conv2d-84           [-1, 32, 96, 96]          27,680
      BatchNorm2d-85           [-1, 32, 96, 96]              64
             ReLU-86           [-1, 32, 96, 96]               0
           Conv2d-87           [-1, 32, 96, 96]           9,248
      BatchNorm2d-88           [-1, 32, 96, 96]              64
             ReLU-89           [-1, 32, 96, 96]               0
       DoubleConv-90           [-1, 32, 96, 96]               0
          UpBlock-91           [-1, 32, 96, 96]               0
           Conv2d-92            [-1, 1, 96, 96]              33
         Upsample-93          [-1, 512, 12, 12]               0
           Conv2d-94          [-1, 256, 12, 12]       1,769,728
      BatchNorm2d-95          [-1, 256, 12, 12]             512
             ReLU-96          [-1, 256, 12, 12]               0
           Conv2d-97          [-1, 256, 12, 12]         590,080
      BatchNorm2d-98          [-1, 256, 12, 12]             512
             ReLU-99          [-1, 256, 12, 12]               0
      DoubleConv-100          [-1, 256, 12, 12]               0
         UpBlock-101          [-1, 256, 12, 12]               0
        Upsample-102          [-1, 256, 24, 24]               0
          Conv2d-103          [-1, 128, 24, 24]         442,496
     BatchNorm2d-104          [-1, 128, 24, 24]             256
            ReLU-105          [-1, 128, 24, 24]               0
          Conv2d-106          [-1, 128, 24, 24]         147,584
     BatchNorm2d-107          [-1, 128, 24, 24]             256
            ReLU-108          [-1, 128, 24, 24]               0
      DoubleConv-109          [-1, 128, 24, 24]               0
         UpBlock-110          [-1, 128, 24, 24]               0
        Upsample-111          [-1, 128, 48, 48]               0
          Conv2d-112           [-1, 64, 48, 48]         110,656
     BatchNorm2d-113           [-1, 64, 48, 48]             128
            ReLU-114           [-1, 64, 48, 48]               0
          Conv2d-115           [-1, 64, 48, 48]          36,928
     BatchNorm2d-116           [-1, 64, 48, 48]             128
            ReLU-117           [-1, 64, 48, 48]               0
      DoubleConv-118           [-1, 64, 48, 48]               0
         UpBlock-119           [-1, 64, 48, 48]               0
        Upsample-120           [-1, 64, 96, 96]               0
          Conv2d-121           [-1, 32, 96, 96]          27,680
     BatchNorm2d-122           [-1, 32, 96, 96]              64
            ReLU-123           [-1, 32, 96, 96]               0
          Conv2d-124           [-1, 32, 96, 96]           9,248
     BatchNorm2d-125           [-1, 32, 96, 96]              64
            ReLU-126           [-1, 32, 96, 96]               0
      DoubleConv-127           [-1, 32, 96, 96]               0
         UpBlock-128           [-1, 32, 96, 96]               0
          Conv2d-129            [-1, 1, 96, 96]              33
----------------------------------------------------------------

