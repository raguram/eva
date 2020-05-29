import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from cnnlib import Utility


class ModelSummaryWriter:

    def __init__(self, name, num_images=5, fig_size=(15, 15)):
        self.num_images = num_images
        self.fig_size = fig_size
        self.writer = SummaryWriter(comment=name)

    def write_pred_summary(self, data, mask, depth):
        figure = plt.figure(figsize=self.fig_size)
        rows = self.num_images
        cols = len(mask[:self.num_images])

        def plt_images(images, start_idx):
            for idx, img in enumerate(images):
                img = img.to(Utility.getCpu())
                plt.subplot(rows, cols, start_idx + idx)
                plt.axis("off")
                plt.imshow(np.asarray(img.squeeze()), cmap='gray')
            return start_idx + len(images)

        start_idx = plt_images(data['fg_bg'][:self.num_images].permute(0, 2, 3, 1), 1)
        start_idx = plt_images(data['fg_bg_mask'][:self.num_images].permute(0, 2, 3, 1), start_idx)
        start_idx = plt_images(data['fg_bg_depth'][:self.num_images].permute(0, 2, 3, 1), start_idx)
        start_idx = plt_images(mask[:self.num_images].permute(0, 2, 3, 1), start_idx)
        plt_images(depth[:self.num_images].permute(0, 2, 3, 1), start_idx)

        self.writer.add_figure("batch sample", figure=figure)

    def write_loss_summary(self, metric_name, loss, global_step):
        self.writer.add_scalar(metric_name, loss, global_step)
