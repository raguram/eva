# Eva - Foreground Object Segmentation & Depth Estimation

## Goal

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

### Dataset creation 

Details of the dataset can be seen [here](https://github.com/raguram/eva/blob/master/S15/documentation/Dataset-Creation.md)

### Implementation Details 

Implementation details is detailed [here](https://github.com/raguram/eva/blob/master/S15/documentation/Design.md)

### Results 

Results are highlighted in this readme file [here](https://github.com/raguram/eva/blob/master/S15/documentation/Results.md)


