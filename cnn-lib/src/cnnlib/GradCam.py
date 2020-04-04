import torch
import torch.nn.functional as F
from cnnlib import Utility
import cv2
import numpy as np
from matplotlib import pyplot as plt


class GradCam:

    def __init__(self, model, target_layers):

        self.model = model
        self.target_layers = target_layers
        self.feature_map = {}
        self.gradient_map = {}
        self._register_hooks()

    def _create_forward_hook(self, layer, module):
        def forward_hook(module, input, output):
            self.feature_map[layer] = output.detach()

        return forward_hook

    def _create_backward_hook(self, layer, module):
        def backward_hook(module, grad_input, grad_output):
            self.gradient_map[layer] = grad_output[0].detach()

        return backward_hook

    def _register_hooks(self):

        for layer, module in self.model.named_modules():
            if layer in self.target_layers:
                module.register_forward_hook(self._create_forward_hook(layer, module))
                module.register_backward_hook(self._create_backward_hook(layer, module))

    def _create_one_hots(self, output, pred):
        one_hots = torch.zeros_like(output)
        for i, one_hot in enumerate(one_hots):
            one_hot[pred[i]] = 1.0
        return one_hots

    def _forward(self, data):
        output = self.model(data)
        pred = output.argmax(dim=1, keepdim=True)
        return output, pred

    def _backward(self, output, pred):

        one_hots = self._create_one_hots(output, pred)
        self.model.zero_grad()
        output.backward(gradient=one_hots)

    def __call__(self, data):
        output, pred = self._forward(data)
        self._backward(output, pred=pred)

        heat_maps = {}
        batch_size, channels, height, width = data.shape
        # for each of the target layers
        for layer in self.target_layers:
            feature_map = self.feature_map[layer]
            gradient_map = self.gradient_map[layer]
            weights = F.adaptive_avg_pool2d(gradient_map, 1)

            heat_map = torch.mul(feature_map, weights).sum(dim=1, keepdim=True)
            heat_map = F.relu(heat_map)

            # Bilinear interpolate the map to bigger dimension
            heat_map = F.interpolate(heat_map, size=(height, width), mode="bilinear")

            heat_map = heat_map.view(batch_size, -1)
            heat_map -= heat_map.min(dim=1, keepdim=True).values
            heat_map /= heat_map.max(dim=1, keepdim=True).values + 0.0000000001
            heat_map = heat_map.view(batch_size, -1, height, width)

            heat_maps[layer] = heat_map

        return heat_maps, pred


class Analyzer:

    def __init__(self, gradCam):

        self.gradCam = gradCam

    def visualize(self, data, data_targets, classes, count=5, muSigPair=None):

        heatmaps, cam_pred = self.gradCam(data)
        randIndices = Utility.pickRandomElements(data, count)

        rand_data, rand_targets, rand_cam_pred = data[randIndices], data_targets[randIndices], cam_pred[randIndices]
        rand_superImposedImages = {}
        for layer in heatmaps:
            rand_superImposedImages[layer] = self.superImpose(rand_data, heatmaps[layer], muSigPair)

        self.plot(Utility.toImages(rand_data, muSigPair), rand_targets, rand_cam_pred, rand_superImposedImages, classes)

    def plot(self, images, targets, preds, layerImagesMap, classes):

        c = len(layerImagesMap) + 2
        r = len(images) + 1

        fig = plt.figure(figsize=(10, 5))
        fig.subplots_adjust(hspace=0.01, wspace=0.01)

        # Print 'Original' in the first row, second column
        ax = plt.subplot(r, c, 2)
        ax.text(0.3, 0, "ORIGINAL")
        plt.axis('off')

        # Print the layer names in the first row.
        for j, layer in enumerate(layerImagesMap):
            ax = plt.subplot(r, c, j + 3)
            ax.text(0.3, 0, layer.upper())
            plt.axis('off')

        # Show the images
        for i in range(len(images)):

            # Print the prediction and the target class in the first column of each row.
            ax = plt.subplot(r, c, (i + 1) * c + 1)
            ax.text(-0.3, 0.5, f"pred={classes[preds[i]]}\n[actual={classes[targets[i]]}]")
            plt.axis('off')

            # Show the original image
            plt.subplot(r, c, (i + 1) * c + 2)
            plt.imshow(images[i], interpolation='bilinear')
            plt.axis('off')

            # Show the layer superimposed images
            for j, layer in enumerate(layerImagesMap):
                plt.subplot(r, c, (i + 1) * c + j + 3)
                plt.imshow(layerImagesMap[layer][i], interpolation='bilinear')
                plt.axis('off')

        plt.show()

    def superImpose(self, data, heatMapImages, muSigPair):

        superImposedImages = []
        images = Utility.toImages(data, muSigPair)
        for i, image in enumerate(images):
            image = np.uint8(255 * image)
            heatmap = 1 - heatMapImages[i]
            heatmap = np.uint8(255 * heatmap.squeeze())
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superImposed = cv2.addWeighted(image, 0.5, heatmap, 0.5, 0)
            superImposedImages.append(superImposed)

        return superImposedImages
