import math
from matplotlib import pyplot as plt


class CyclicValueGenerator:

    def __init__(self, min, max, step_size):

        if (step_size <= 0):
            raise ValueError("Step size should be greater than 0")
        self.min = min
        self.max = max
        self.step_size = step_size

    """
    Get the value for the given iteration.
    """

    def getValue(self, iteration):
        completed_steps = math.floor(iteration / self.step_size)
        iterationInCurrentStep = iteration % self.step_size

        ratio = iterationInCurrentStep / self.step_size
        if (completed_steps % 2 == 1):
            ratio = 1 - ratio

        return self.min + (self.max - self.min) * ratio

    def getValues(self, numberOfIterations):

        values = [self.getValue(i) for i in range(0, numberOfIterations)]
        return values

    def plotValues(self, numberOfIterations, fig_size=(10, 5)):
        values = self.getValues(numberOfIterations)
        plt.figure(figsize=fig_size)
        plt.plot(values)
        plt.xlabel("Iteration")
        plt.ylabel("Value")
        plt.show()

# CyclicValueGenerator(1, 10, step_size=10).plotValues(100)
