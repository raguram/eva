## Problem Statement 

The goal of this assignment is to train a MNIST model having validation accuracy of atleast 99.4% with the following constraints. 

* 99.4% validation accuracy
* Less than 10k Parameters
* Less than or equal to 15 Epochs

## Iteration 1

#### Target
Goal of this is to get the setup done. With basic trasnformers, train, test and accuracy plots. This is to ensure that code I have written works correctly.

#### Result
* Parameters = 6,379,786
* Best train accuracy = 99.97%
* Best validation accuracy = 99.29 (12th epoch)

#### Analysis
* Huge difference between the train and the test accuracy indicating the model is overfitting.
* Very high number of parameters resuling in very heavy model.

#### File
https://github.com/raguram/eva/blob/master/S5/0-MNIST-Setup.ipynb

## Iteration 2

#### Target
Goal of this is to come up with a skeleton having a very light model. This will act as a baseline for my training. In the next iterations, I will be able to improve in many dimensions like its complexity, efficiency, accuracy etc.

#### Result
* Parameters = 5,416
* Best train accuracy = 98.40166666666667%
* Best validation accuracy = 98.34%

#### Analysis
* Model's train and validation accuracy are around the same indicating the model has NOT overfit.
* With higher epochs, the model accuracy was still improving. Looks like if we increase the efficiency of the model, we can push this model further with the same number of parameters and epochs.

#### File
https://github.com/raguram/eva/blob/master/S5/1-MNIST-BaseModel.ipynb

## Iteration 3

#### Target
Goal of this is to push the base line architecture by improving the efficiency. I used batch normalization. This is because, in my previous iteration, I noticed that increasing the number of epochs, resulted in slight increase in the model accuracy.

#### Result
* Parameters = 5,512
* Best train accuracy = 98.69%
* Best validation accuracy = 98.47%

#### Analysis
* Model training accuracy has saturated and it does not seem to increase beyond 98.69% even if I tried increasing the number of epochs, indicating that the model is underfitting and not complex enough.
* Also, the train and the test accuracies are also closer. No signs of overfitting to the train data.

#### File
https://github.com/raguram/eva/blob/master/S5/2-MNIST-BatchNorm.ipynb

## Iteration 4 

#### Target
In the previous iteration, the base line architecture with 5,512 parameters saturated at 98.69%. So, the goal here is to increase the complexity of the model by increasing the number of parameters. This should improve the accuracy.

#### Result
* Parameters = 9,638
* Best train accuracy = 99.303%
* Best validation accuracy = 98.81% (12th epoch)

#### Analysis
* Model is clearly overfitting indicated by the difference between the model training accuracy and the validation accuracy.
* Validation accuracy has marginally increased indicating that in the next iteration if I use appropriate regularization, I will be able to push the accuracy further.

#### File
https://github.com/raguram/eva/blob/master/S5/3-MNIST-IncreasedModelComplexity.ipynb

## Iteration 5 

#### Target
The goal of this iteration is to avoid overfitting happening in the previous iteration by introducing drop outs.

#### Result
* Parameters = 9,638
* Best train accuracy = 98.79%
* Best validation accuracy = 99.14%

#### Analysis
* Model is not overfitting.
* Also, increasing the number of epochs did not give better result. So, we have to further improve the capacity of the model to achieve the required result.

#### File
https://github.com/raguram/eva/blob/master/S5/4-MNIST-DropOutRegularized.ipynb

## Iteration 6 

#### Target
Target is to increase the capacity of the model by tweaking the layers a bit. I added more layers towards the end motivated by the EVA4S5F8 model architecture.

#### Result
* Parameters = 9,876
* Best train accuracy = 99.35%
* Best validation accuracy = 99.41%

#### Analysis
* Model seems to perform well. However, there is room for improvement as I am not seeing the required accuracy consistently.

#### File
https://github.com/raguram/eva/blob/master/S5/5-MNIST-ImprovedModelArchitecture.ipynb

## Iteration 7 

#### Target
Based on the misclassifications, the goal is to increase the accuracy by trying data transforms.

#### Result
* Parameters = 9,876
* Best train accuracy = 98.94%
* Best validation accuracy = 99.45%

#### Analysis
* Though the test accuracy was greater than 99.4% for 4 epochs, from the graph it can be seen that the accuracy is fluctuating. We still have to improve the model to stabilize the same across epochs.

#### File
https://github.com/raguram/eva/blob/master/S5/6-MNIST-DataTransforms.ipynb

## Iteration 8 

#### Target
In the previous iteration, test accuracy was fluctuating post 10 epochs. So, the goal of this is to try and reduce the LR post 10 epochs to make sure the model stabilizes.

#### Result
* Parameters = 9,876
* Best train accuracy = 99.073%
* Best validation accuracy = 99.45%

#### Analysis
* The model looks to be fairly stable after 10 epochs.
* Train accuracy is well below the validation accuracy because the transformations added made the training set complex.

#### File
https://github.com/raguram/eva/blob/master/S5/7-MNIST-StepLR.ipynb
