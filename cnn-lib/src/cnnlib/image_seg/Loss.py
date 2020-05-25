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
