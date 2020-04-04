from cnnlib.ModelBuilder import ModelBuildResult
from cnnlib import MetricsUtility

trainAcc = [45.876, 63.8, 70.662, 75.02, 77.5, 79.52, 81.056, 82.608, 83.394, 84.164, 85.192, 85.616, 86.438, 87.2, 87.56, 88.126, 88.924, 89.012, 89.538, 89.816, 90.216, 90.782, 90.842, 91.316, 91.452, 91.66, 92.078, 92.092, 92.42, 92.468, 92.778, 92.812, 93.202, 93.232, 93.484, 93.57, 93.928, 93.722, 94.192, 94.398]
testAcc = [60.39, 70.91, 73.83, 75.09, 78.87, 80.45, 83.33, 84.26, 85.38, 85.32, 84.61, 87.2, 87.11, 87.02, 87.99, 88.91, 87.82, 87.67, 88.6, 85.78, 88.79, 89.23, 88.42, 89.22, 90.21, 90.19, 89.58, 88.47, 90.44, 91.24, 90.26, 90.84, 89.28, 91.26, 90.53, 90.22, 91.57, 91.58, 91.45, 91.58]

result = ModelBuildResult(trainAccuracies=trainAcc, testAccuracies=testAcc, trainLosses=[], testLosses=[])
MetricsUtility.plotMetrics(result)