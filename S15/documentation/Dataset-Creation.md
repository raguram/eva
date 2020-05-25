## Dataset

### Dataset Creation

#### Step 1 - Download background images (bg) from google 
Downloaded ~100 background images from google images using the [plugin](https://chrome.google.com/webstore/detail/fatkun-batch-download-ima/nnjjahlikiabnchcpehcpkdeckfgnohf?hl=en). 

![background-sample](https://github.com/raguram/eva/blob/master/S15/ReadMeImages/bg.png)

#### Step 2 - Download foreground (fg) images and removed the background to make it transparent. 
Similarly to background, picked up ~100 foreground images from google. For each of the foreground images, used [background remover](https://www.remove.bg/) to remove the background and create transparent foreground images (png). 

![foreground-sample](https://github.com/raguram/eva/blob/master/S15/ReadMeImages/fg.png)

#### Step 3 - Preprocess the fg and bg images. 
* Reshaped the background images to (224, 224, 3) creating 'bg'.
* Thumbnailed the foreground images to (150, 150, 3) creating 'fg'.
* Appended the mirrors of each of the foreground images to the 'fg'. 

![background-processed](https://github.com/raguram/eva/blob/master/S15/ReadMeImages/bg_processed.png)
![foreground-processed](https://github.com/raguram/eva/blob/master/S15/ReadMeImages/fg_processed.png)

#### Step 4 - fg_mask: Create mask for fg images 
For each of the images in 'fg', created mask by using the alpha channel values and creating (150, 150, 1) image.

![foreground-mask](https://github.com/raguram/eva/blob/master/S15/ReadMeImages/fg_mask.png)

#### Step 5 - fg_bg: Paste each the fg images on each of the bg images at 20 random positions 
Used pillow paste to overlay fg on top of bg image at 20 different positions to create fg_bg. 

![fg-bg](https://github.com/raguram/eva/blob/master/S15/ReadMeImages/bg_fg.png)

#### Step 5 - fg_bg_mask: Paste each of the images in fg_mask to same 20 random positions in bg
Similar to fg_bg creation, used pillow paste. 
![fg-bg-mask](https://github.com/raguram/eva/blob/master/S15/ReadMeImages/bg_fg_mask.png)

#### Step 6 - fg_bg_depth: For each of the fg_bg images created a depth map
Modified [Depth Models](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb) to generate depth map for each of the fg_bg images. 

![fg-bg](https://github.com/raguram/eva/blob/master/S15/ReadMeImages/depth_1.png)
![fg-bg-depth](https://github.com/raguram/eva/blob/master/S15/ReadMeImages/depth_output.png)

### Data Statistics 

- Dataset location: [data.zip](https://drive.google.com/open?id=1NL7ZwDcC0P64L2n_LWqlQSJB45lAHpfD)
- Zip size: 8 GB 

| Folder Name | # of images |  Dimension  |          Mean          |          Std           |
|-------------|-------------|-------------|------------------------|------------------------|
| bg          |         110 | 3, 224, 224 | 0.4491, 0.4220, 0.3982 | 0.0056, 0.0045, 0.0048 |
| fg          |         208 | 3, w, h     |                        |                        |
| fg_bg       |      457600 | 3, 224, 224 | 0.4393, 0.4124, 0.3921 | 0.0041, 0.0036, 0.0042 |
| fg_bg_mask  |      457600 | 1, 224, 224 | 0.0846                 | 0.0432                 |
| fg_bg_depth |      457600 | 1, 224, 224 | 0.3870                 | 0.0035                 |

