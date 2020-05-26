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

To start with, I created a tiny dataset from of the bigger set containing randomly picked up 20k fg_bg images with their corresponding mask and depth images. The tiny data set can be accessed at [tiny_data.zip](https://drive.google.com/open?id=1Tw2Ijf2l7fERsOEQ7k_IMRXYTzm4BaI4). All of the analysis of the below losses were performed on this dataset. 
I created an experiment suite, where it gets quicker to conduct experiments on this dataset. Code can be found [here](https://github.com/raguram/eva/blob/master/S15/Experiment-Suite.ipynb). 

During the literature survey, I came across multiple losses. I picked up few relevant ones and performed an analysis. The following section gives an overview of the losses that I tried and corresponding results and observation. 

#### BCEWithLogits

This loss is a combination of sigmoid and Binary Cross Entropy. Leaving the formulations aside, logically, it first applies a sigmoid on to the output to bring the output in range (0, 1). Then, for every pixel in the output image, computes the binary cross entropy with the corresponding pixel in the target. More details can be found [here](https://pytorch.org/docs/master/generated/torch.nn.BCEWithLogitsLoss.html)

BCE made sense to be applied on mask image as the target pixel values 0s or 1s. 

- Loss: BCEWithLogistisLoss
- Optimizer: SGD
- LR: 0.5
- Momentum: 0.9
- Epochs: 20
- Experiment [link](https://github.com/raguram/eva/blob/master/S15/Experiment-Suite.ipynb) 

![FG_BG](https://github.com/raguram/eva/blob/master/S15/documentation/BCE_Mask_FG_BG.png)
![FG_BG_MASK](https://github.com/raguram/eva/blob/master/S15/documentation/BCE_MASK_FG_BG_MASK.png)
![FG_BG_PREDICTED](https://github.com/raguram/eva/blob/master/S15/documentation/BCE_MASK_FG_BG_MASK_PRED.png)

The convergence was slow and the I guess the class imbalance (number of white pixels are low), might have caused the predicted image to close to a negative of the actual truth data. I am yet to identify the root cause of why this loss function is not appropriate. 

#### L1 Loss 

L1 loss is mean absolute error betwen each of the pixels in the output and the prediction. 

- Loss: BCEWithLogistisLoss
- Optimizer: SGD
- LR: 0.5
- Momentum: 0.9
- Epochs: 20
- Experiment [link](https://github.com/raguram/eva/blob/master/S15/Experiment-Suite.ipynb) 

![FG_BG](https://github.com/raguram/eva/blob/master/S15/documentation/L1_FG_BG.png)
![FG_BG_MASK](https://github.com/raguram/eva/blob/master/S15/documentation/L1_FG_BG_MASK.png)
![FG_BG_PREDICTED](https://github.com/raguram/eva/blob/master/S15/documentation/L1_FG_BG_PRED.png)

#### Root Mean Squared Error Loss 

This is L2 loss taking the squared difference between the values of the pixels. 

#### Dice Loss 

This loss is similar to IOU. 

I came across other losses. Some of them are - Huber loss - Loss which combines the L1 and L2, where the loss is linear for large values and quadratic for smaller values, SSIM - Structural Similarity Index, Tversky loss etc. In the interest of time, to avoid going into analysis paralysis, I decided to stick to combinations of the above loss functions and tried to get the juice out of the network.

## Implementation Detail

In this section, I will talk about the implementation details of the project. 

### Low Level Design 

[TODO Add details]

- [Cnn-lib](https://github.com/raguram/eva/tree/master/cnn-lib)
- [Custom data set](https://github.com/raguram/eva/blob/master/cnn-lib/src/cnnlib/datasets/DepthDataset.py) 
- [Model builder](https://github.com/raguram/eva/blob/master/cnn-lib/src/cnnlib/image_seg/ModelBuilder.py) 
- [Output Persister](https://github.com/raguram/eva/blob/master/cnn-lib/src/cnnlib/image_seg/PredictionPersister.py) 
- [Loss](https://github.com/raguram/eva/blob/master/cnn-lib/src/cnnlib/image_seg/Loss.py) 

### Optimization Bug Fixes

[TODO Add details]
- Torch cache cleanup 
- Deleting the tensors for cleaning up memory 

## Appendix 

#### Summary of the architecture 

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1         [-1, 32, 224, 224]             192
       BatchNorm2d-2         [-1, 32, 224, 224]              64
            Conv2d-3         [-1, 32, 224, 224]           1,760
       BatchNorm2d-4         [-1, 32, 224, 224]              64
              ReLU-5         [-1, 32, 224, 224]               0
            Conv2d-6         [-1, 32, 224, 224]           9,248
       BatchNorm2d-7         [-1, 32, 224, 224]              64
              ReLU-8         [-1, 32, 224, 224]               0
        DoubleConv-9         [-1, 32, 224, 224]               0
             ReLU-10         [-1, 32, 224, 224]               0
        MaxPool2d-11         [-1, 32, 112, 112]               0
         ResBlock-12  [[-1, 32, 112, 112], [-1, 32, 224, 224]]               0
           Conv2d-13         [-1, 64, 112, 112]           2,048
      BatchNorm2d-14         [-1, 64, 112, 112]             128
           Conv2d-15         [-1, 64, 112, 112]          18,496
      BatchNorm2d-16         [-1, 64, 112, 112]             128
             ReLU-17         [-1, 64, 112, 112]               0
           Conv2d-18         [-1, 64, 112, 112]          36,928
      BatchNorm2d-19         [-1, 64, 112, 112]             128
             ReLU-20         [-1, 64, 112, 112]               0
       DoubleConv-21         [-1, 64, 112, 112]               0
             ReLU-22         [-1, 64, 112, 112]               0
        MaxPool2d-23           [-1, 64, 56, 56]               0
         ResBlock-24  [[-1, 64, 56, 56], [-1, 64, 112, 112]]               0
           Conv2d-25          [-1, 128, 56, 56]           8,192
      BatchNorm2d-26          [-1, 128, 56, 56]             256
           Conv2d-27          [-1, 128, 56, 56]          73,856
      BatchNorm2d-28          [-1, 128, 56, 56]             256
             ReLU-29          [-1, 128, 56, 56]               0
           Conv2d-30          [-1, 128, 56, 56]         147,584
      BatchNorm2d-31          [-1, 128, 56, 56]             256
             ReLU-32          [-1, 128, 56, 56]               0
       DoubleConv-33          [-1, 128, 56, 56]               0
             ReLU-34          [-1, 128, 56, 56]               0
        MaxPool2d-35          [-1, 128, 28, 28]               0
         ResBlock-36  [[-1, 128, 28, 28], [-1, 128, 56, 56]]               0
           Conv2d-37          [-1, 256, 28, 28]          32,768
      BatchNorm2d-38          [-1, 256, 28, 28]             512
           Conv2d-39          [-1, 256, 28, 28]         295,168
      BatchNorm2d-40          [-1, 256, 28, 28]             512
             ReLU-41          [-1, 256, 28, 28]               0
           Conv2d-42          [-1, 256, 28, 28]         590,080
      BatchNorm2d-43          [-1, 256, 28, 28]             512
             ReLU-44          [-1, 256, 28, 28]               0
       DoubleConv-45          [-1, 256, 28, 28]               0
             ReLU-46          [-1, 256, 28, 28]               0
        MaxPool2d-47          [-1, 256, 14, 14]               0
         ResBlock-48  [[-1, 256, 14, 14], [-1, 256, 28, 28]]               0
           Conv2d-49          [-1, 512, 14, 14]       1,180,160
      BatchNorm2d-50          [-1, 512, 14, 14]           1,024
             ReLU-51          [-1, 512, 14, 14]               0
           Conv2d-52          [-1, 512, 14, 14]       2,359,808
      BatchNorm2d-53          [-1, 512, 14, 14]           1,024
             ReLU-54          [-1, 512, 14, 14]               0
       DoubleConv-55          [-1, 512, 14, 14]               0
         Upsample-56          [-1, 512, 28, 28]               0
           Conv2d-57          [-1, 256, 28, 28]       1,769,728
      BatchNorm2d-58          [-1, 256, 28, 28]             512
             ReLU-59          [-1, 256, 28, 28]               0
           Conv2d-60          [-1, 256, 28, 28]         590,080
      BatchNorm2d-61          [-1, 256, 28, 28]             512
             ReLU-62          [-1, 256, 28, 28]               0
       DoubleConv-63          [-1, 256, 28, 28]               0
          UpBlock-64          [-1, 256, 28, 28]               0
         Upsample-65          [-1, 256, 56, 56]               0
           Conv2d-66          [-1, 128, 56, 56]         442,496
      BatchNorm2d-67          [-1, 128, 56, 56]             256
             ReLU-68          [-1, 128, 56, 56]               0
           Conv2d-69          [-1, 128, 56, 56]         147,584
      BatchNorm2d-70          [-1, 128, 56, 56]             256
             ReLU-71          [-1, 128, 56, 56]               0
       DoubleConv-72          [-1, 128, 56, 56]               0
          UpBlock-73          [-1, 128, 56, 56]               0
         Upsample-74        [-1, 128, 112, 112]               0
           Conv2d-75         [-1, 64, 112, 112]         110,656
      BatchNorm2d-76         [-1, 64, 112, 112]             128
             ReLU-77         [-1, 64, 112, 112]               0
           Conv2d-78         [-1, 64, 112, 112]          36,928
      BatchNorm2d-79         [-1, 64, 112, 112]             128
             ReLU-80         [-1, 64, 112, 112]               0
       DoubleConv-81         [-1, 64, 112, 112]               0
          UpBlock-82         [-1, 64, 112, 112]               0
         Upsample-83         [-1, 64, 224, 224]               0
           Conv2d-84         [-1, 32, 224, 224]          27,680
      BatchNorm2d-85         [-1, 32, 224, 224]              64
             ReLU-86         [-1, 32, 224, 224]               0
           Conv2d-87         [-1, 32, 224, 224]           9,248
      BatchNorm2d-88         [-1, 32, 224, 224]              64
             ReLU-89         [-1, 32, 224, 224]               0
       DoubleConv-90         [-1, 32, 224, 224]               0
          UpBlock-91         [-1, 32, 224, 224]               0
           Conv2d-92          [-1, 1, 224, 224]              33
         Upsample-93          [-1, 512, 28, 28]               0
           Conv2d-94          [-1, 256, 28, 28]       1,769,728
      BatchNorm2d-95          [-1, 256, 28, 28]             512
             ReLU-96          [-1, 256, 28, 28]               0
           Conv2d-97          [-1, 256, 28, 28]         590,080
      BatchNorm2d-98          [-1, 256, 28, 28]             512
             ReLU-99          [-1, 256, 28, 28]               0
      DoubleConv-100          [-1, 256, 28, 28]               0
         UpBlock-101          [-1, 256, 28, 28]               0
        Upsample-102          [-1, 256, 56, 56]               0
          Conv2d-103          [-1, 128, 56, 56]         442,496
     BatchNorm2d-104          [-1, 128, 56, 56]             256
            ReLU-105          [-1, 128, 56, 56]               0
          Conv2d-106          [-1, 128, 56, 56]         147,584
     BatchNorm2d-107          [-1, 128, 56, 56]             256
            ReLU-108          [-1, 128, 56, 56]               0
      DoubleConv-109          [-1, 128, 56, 56]               0
         UpBlock-110          [-1, 128, 56, 56]               0
        Upsample-111        [-1, 128, 112, 112]               0
          Conv2d-112         [-1, 64, 112, 112]         110,656
     BatchNorm2d-113         [-1, 64, 112, 112]             128
            ReLU-114         [-1, 64, 112, 112]               0
          Conv2d-115         [-1, 64, 112, 112]          36,928
     BatchNorm2d-116         [-1, 64, 112, 112]             128
            ReLU-117         [-1, 64, 112, 112]               0
      DoubleConv-118         [-1, 64, 112, 112]               0
         UpBlock-119         [-1, 64, 112, 112]               0
        Upsample-120         [-1, 64, 224, 224]               0
          Conv2d-121         [-1, 32, 224, 224]          27,680
     BatchNorm2d-122         [-1, 32, 224, 224]              64
            ReLU-123         [-1, 32, 224, 224]               0
          Conv2d-124         [-1, 32, 224, 224]           9,248
     BatchNorm2d-125         [-1, 32, 224, 224]              64
            ReLU-126         [-1, 32, 224, 224]               0
      DoubleConv-127         [-1, 32, 224, 224]               0
         UpBlock-128         [-1, 32, 224, 224]               0
          Conv2d-129          [-1, 1, 224, 224]              33
----------------------------------------------------------------
