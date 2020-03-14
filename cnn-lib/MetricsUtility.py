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
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs[0, 0].plot(modelBuildResult.testAccuracies)
    axs[0, 0].set_title("Training Accuracy")
    axs[0, 1].plot(modelBuildResult.trainLosses)
    axs[0, 1].set_title("Training Loss")
    axs[1, 0].plot(modelBuildResult.testAccuracies)
    axs[1, 0].set_title("Test Accuracy")
    axs[1, 1].plot(modelBuildResult.testLosses)
    axs[1, 1].set_title("Test Loss")
