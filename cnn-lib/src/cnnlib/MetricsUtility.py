from sklearn.metrics import confusion_matrix
import torch
from matplotlib import pyplot as plt

class Metrics:

    def __init__(self, accuracy, data, predictions, targets):
        self.accuracy = accuracy
        self.miss_classified_data = data
        self.miss_classified_predictions = predictions
        self.miss_classified_targets = targets


cpu = torch.device("cpu")


def compute_accuracy(prediction, target):
    return 100 * prediction.eq(target.view_as(prediction)).sum().item() / float(len(prediction))


def computeMetrics(data, prediction, target):
    """
    Computes the accuracy metrics and returns miss classified information
    """
    corrects = prediction.eq(target.view_as(prediction))
    accuracy = compute_accuracy(prediction, target)
    miss_indices = ~corrects
    miss_data = data[miss_indices]
    miss_predictions = prediction[miss_indices]
    miss_targets = target[miss_indices]
    print(confusion_matrix(target.to(cpu), prediction.to(cpu)))
    return Metrics(accuracy, miss_data, miss_predictions, miss_targets)


def plotMetrics(modelBuildResult):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].plot(modelBuildResult.trainAccuracies, label="Train")
    axs[0].plot(modelBuildResult.testAccuracies, label="Test")
    axs[0].set_title("Accuracy")
    axs[0].legend()

    axs[1].plot(modelBuildResult.trainLosses, label="Train")
    axs[1].plot(modelBuildResult.testLosses, label="Test")
    axs[1].set_title("Loss")
    axs[1].legend()

    axs[2].plot(modelBuildResult.learningRates, label="Learning Rate")
    axs[2].set_title("Learning Rate")

