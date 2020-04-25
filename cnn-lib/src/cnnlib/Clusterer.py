import json
import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
import matplotlib.cm as cm


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
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=0)
        kmeans.fit(self.points)
        centroids = kmeans.cluster_centers_
        lables = kmeans.labels_
        meanIOU = sum(
            [self.__compute_IOU__(centroids[lables[i]][0], centroids[lables[i]][1], self.points[i][0],
                                  self.points[i][1]) for i in
             range(len(self.points))]) / len(self.points)

        return (kmeans.labels_, kmeans.cluster_centers_, meanIOU)

    def fit(self, max_clusters):
        self.ious = []
        self.all_labels = []
        self.all_centroids = []
        for k in range(1, max_clusters):
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

        x = np.arange(max(ks))
        ys = [i + x + (i * x) ** 2 for i in range(max(ks))]
        colors = cm.rainbow(np.linspace(0, 1, len(ys)))

        if len(ks) == 1:
            map = self.__labels_points_map(ks[0])
            axs.set_title(f"Max_Clusters = {ks[0]}, Actual_clusters = {len(map)}")
            for label in map:
                self.__plot_one_label(axs, label, self.all_centroids[ks[0] - 1], map, colors)

        else:
            for index, k in enumerate(ks):
                map = self.__labels_points_map(k)
                axs[index].set_title(f"Max_Clusters = {k}, Actual_clusters = {len(map)}")
                for label in map:
                    self.__plot_one_label(axs[index], label, self.all_centroids[k - 1], map, colors)

    def __plot_one_label(self, axs, label, centroids, map, colors):
        point_xs = [p[0] for p in map[label]]
        point_ys = [p[1] for p in map[label]]
        centroid_x = [centroids[label][0]]
        centroid_y = [centroids[label][1]]
        axs.scatter(point_xs, point_ys, color=colors[label])
        axs.scatter(centroid_x, centroid_y, color=colors[label], marker='^')

    def __labels_points_map(self, k):

        labels = self.all_labels[k - 1]
        labelsPointsMap = {}
        for i in range(len(self.points)):
            if labelsPointsMap.__contains__(labels[i]):
                labelsPointsMap[labels[i]].append(self.points[i])
            else:
                labelsPointsMap[labels[i]] = []
                labelsPointsMap[labels[i]].append(self.points[i])
        return labelsPointsMap

# c = TemplateIdentifier("data/annotations.json")
# c.fit(20)
# c.show_points_centroids([5, 10], figsize=(5, 5))
