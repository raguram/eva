from cnnlib.GradCam import GradCam
from torchsummary import summary
import torchvision.models as models
from albumentations import Compose
from albumentations.pytorch import *
from cnnlib import DataUtility
from cnnlib.DataUtility import Alb
from cnnlib import Utility
import json
from cnnlib.GradCam import Analyzer


def getImageNetClasses():
    class_idx = json.load(open("src/cnnlib/models/imagenet_class_index.json"))
    classes = [class_idx[str(i)][1] for i in range(len(class_idx))]
    return classes


def main():
    print("Gradcam Test")

    net = models.resnet34(pretrained=True)
    net.to(Utility.getDevice())
    summary(net, input_size=(3, 224, 224))

    classes = getImageNetClasses()

    transforms = Compose([
        ToTensor()
    ])
    loader = DataUtility.loadImages("resources/processed-images", Alb(transforms))

    layers = ["layer4", "layer3", "layer2", "layer1"]

    cam = GradCam(net, layers)
    analyzer = Analyzer(cam)

    d, l = iter(loader).next()
    analyzer.visualize(d, l, classes)


if __name__ == "__main__":
    main()
