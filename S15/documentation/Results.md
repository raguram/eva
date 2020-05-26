# Results

This document talks about the results from multiple trials on the dataset described [here](https://github.com/raguram/eva/blob/master/S15/documentation/Dataset-Creation.md). It highlights the results from all the different configurations with loss functions. 

## Model 1

- [Architecture](https://github.com/raguram/eva/blob/master/S15/documentation/Design.md#design)
- Loss: 1 * L1Loss(out_mask, target_mask) + 1 * L1Loss(out_depth, target_depth)
- Optimizer: SGD
- LR: 0.5
- Momentum: 0.9
- Epochs: 3
- Model: [model.pt](https://drive.google.com/open?id=1BDvHYchn8CQL7d3ZAckU0y5FhXFKayCC) 

##### Truth data

![FG_BG](https://github.com/raguram/eva/blob/master/S15/documentation/L1_L1_FG_BG.png)
![FG_BG_MASK](https://github.com/raguram/eva/blob/master/S15/documentation/L1_L1_FG_BG_MASK.png)
![FG_BG_DEPTH](https://github.com/raguram/eva/blob/master/S15/documentation/L1_L1_FG_BG_DEPTH.png)

##### Predicted Result 

![FG_BG_MASK_PRED](https://github.com/raguram/eva/blob/master/S15/documentation/L1_L1_FG_BG_MASK_PRED.png)
![FG_BG_DEPTH_PRED](https://github.com/raguram/eva/blob/master/S15/documentation/L1_L1_FG_BG_DEPTH_PRED.png) 

##### Observation 

This model is a failure as the mask predictions are mostly complete black or white.

## Model 2

[TODO]
