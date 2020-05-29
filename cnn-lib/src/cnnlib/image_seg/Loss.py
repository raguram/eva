import torch.nn as nn
import torch


class Loss_fn:

    def __init__(self, mask_loss, depth_loss, alpha=None, beta=None):
        self.mask_loss = mask_loss
        self.depth_loss = depth_loss
        self.alpha = alpha
        self.beta = beta

    def __call__(self, out, target):

        if self.depth_loss is None:
            return self.mask_loss(out['fg_bg_mask'], target['fg_bg_mask'])

        if self.mask_loss is None:
            return self.depth_loss(out['fg_bg_depth'], target['fg_bg_depth'])

        return self.alpha * self.mask_loss(out['fg_bg_mask'], target['fg_bg_mask']) + self.beta * self.depth_loss(
            out['fg_bg_depth'], target['fg_bg_depth'])


class RootMeanSquaredErrorLoss(nn.Module):

    def __init__(self, eps=1e-6):
        super().__init__()
        self.loss = nn.MSELoss()
        self.eps = eps

    def forward(self, out, target):
        out = torch.sigmoid(out)
        loss = torch.sqrt(self.loss(out, target) + self.eps)
        return loss


class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.eps: float = 1e-6

    def forward(self, out, target):
        out = torch.sigmoid(out)
        i = torch.sum(out * target, (1, 2, 3))
        u = torch.sum(out + target, (1, 2, 3))
        iou = i / (u + self.eps)
        return torch.mean(torch.tensor(1.0) - 2.0 * iou)

