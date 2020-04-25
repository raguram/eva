import json
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


class TemplateIdentifier:

    def __init__(self, input):
        # Input is going to be the annotations json
        self.input = input
        self.__parse_annotations__()

    def __parse_annotations__(self):
        with open(self.input) as fp:
            annotations = json.load(fp)

        self.img_width = np.array([img['width'] for img in annotations['images']])
        self.img_height = np.array([img['height'] for img in annotations['images']])

        self.bbox_w = np.array([ann['bbox'][2] for ann in annotations['annotations']])
        self.bbox_h = np.array([ann['bbox'][3] for ann in annotations['annotations']])

        self.bbox_h_norm = self.bbox_h / self.img_height
        self.bbox_w_norm = self.bbox_w / self.img_width

        self.points = [(self.bbox_w_norm[i], self.bbox_h_norm[i]) for i in range(len(self.bbox_h_norm))]

    def __compute_IOU__(self, cen_h, cen_w, bbox_h, bbox_w):
        intersection = min(cen_h, bbox_h) * min(cen_w, bbox_w)
        union = cen_h * cen_h + bbox_h * bbox_w - intersection
        return (intersection / union)

    def cluster(self, k):
        kmeans = KMeans(k)
        kmeans.fit(self.points)
        centroids = kmeans.cluster_centers_
        lables = kmeans.labels_
        meanIOU = sum(
            [self.__compute_IOU__(centroids[lables[i]][0], centroids[lables[i]][1], self.points[i][0],
                                  self.points[i][1]) for i in
             range(len(self.points))]) / len(self.points)

        return (kmeans.labels_, kmeans.cluster_centers_, meanIOU)

    def fit(self):
        self.ious = []
        self.all_labels = []
        self.all_centroids = []
        for k in range(1, 20):
            labels, centroids, iou = self.cluster(k)
            self.all_labels.append(labels)
            self.all_centroids.append(centroids)
            self.ious.append(iou)

    def show_iou_curve(self, figsize=(5, 5)):
        fig, axs = plt.subplots(1, 1, figsize)
        axs.plot(self.ious, label="IOUs")
        axs.set_xlabel("Number of Clusters")
        axs.set_ylabel("Mean IOUs")
        axs.legend()

    def show_points_centroids(self, ks, figsize):
        fig, axs = plt.subplots(len(ks), 1, figsize=figsize)
        point_xs = [p[0] for p in self.points]
        point_ys = [p[1] for p in self.points]

        if len(ks) == 1:
            centroids_xs = [c[0] for c in self.all_centroids[ks[0] - 1]]
            centroids_ys = [c[0] for c in self.all_centroids[ks[0] - 1]]
            axs.set_title(f"Clusters = {ks[0]}")
            axs.scatter(point_xs, point_ys)
            axs.scatter(centroids_xs, centroids_ys, c='red')

        else:
            for index, k in enumerate(ks):
                axs[index].scatter(point_xs, point_ys)
                centroids_xs = [c[0] for c in self.all_centroids[ks[index] - 1]]
                centroids_ys = [c[0] for c in self.all_centroids[ks[index] - 1]]
                axs[index].set_title(f"Clusters = {ks[index]}")
                axs[index].scatter(point_xs, point_ys)
                axs[index].scatter(centroids_xs, centroids_ys, c='red')

# c = TemplateIdentifier("data/annotations.json")
# c.fit()
# c.show_points_centroids([5, 10, 15], figsize=(5, 10))
