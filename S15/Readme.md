# Eva - Foreground Object Detection & Depth Estimation

### Goal

Given an image and the background of the image, predict 

* Mask image of the objects in the foreground 
* Predicted depth map of the same.

### Requirements

You must have 100 background, 100x2 (including flip), and you randomly place the foreground on the background 20 times, you have in total 100x200x20 images. 

In total you MUST have:

* 400k fg_bg images
* 400k depth images
* 400k mask images

These are generated from:
* 100 backgrounds
* 100 foregrounds, plus their flips
* 20 random placement on each background.

### Dataset Creation 

#### Step 1 - Download background images (bg) from google 
Downloaded ~100 background images from google images using the [plugin](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en). 

![background-sample](https://github.com/raguram/eva/blob/master/S15/background-sample.png)

#### Step 2 - Download foreground (fg) images and removed the background to make it transparent. 
Similarly to background, picked up ~100 foreground images from google. For each of the foreground images, used [background remover](https://www.remove.bg/) to remove the background and create transparent foreground images (png). 

![foreground-sample](https://github.com/raguram/eva/blob/master/S15/foreground-samples.png)

#### Step 3 - Preprocess the fg and bg images. 
* Reshaped the background images to (224, 224, 3) creating 'bg'.
* Reshaped the foreground images to (150, 150, 3) creating 'fg'.
* Appended the mirrors of each of the foreground images to the 'fg'. 

![background-processed](https://github.com/raguram/eva/blob/master/S15/background-processed.png)
![foreground-processed](https://github.com/raguram/eva/blob/master/S15/foreground-processed.png)

#### Step 4 - fg_mask: Create mask for fg images 
For each of the images in 'fg', created mask by using the alpha channel values and creating (150, 150, 1) image.

![foreground-mask](https://github.com/raguram/eva/blob/master/S15/foreground_mask.png)

#### Step 5 - fg_bg: Paste each the fg images on each of the bg images at 20 random positions 
Used pillow paste to overlay fg on top of bg image at 20 different positions to create fg_bg. 

![fg-bg](https://github.com/raguram/eva/blob/master/S15/fg_bg_samples.png)

#### Step 5 - fg_bg_mask: Paste each of the images in fg_mask to same 20 random positions in bg
Similar to fg_bg creation, used pillow paste. 
![fg-bg-mask](https://github.com/raguram/eva/blob/master/S15/fg_bg_mask.png)

#### Step 6 - fg_bg_depth: For each of the fg_bg images created a depth map
Modified [Depth Models](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb) to generate depth map for each of the fg_bg images. 

