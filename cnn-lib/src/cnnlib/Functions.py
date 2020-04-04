import torch


class LossFn:

    def __init__(self, baseLoss, l1Factor=0.0, l2Factor=0.0):
        self.baseLoss = baseLoss
        self.l1Factor = l1Factor
        self.l2Factor = l2Factor

    def computeLoss(self, model, predictions, targets):
        return self.baseLoss(predictions, targets) + self.l1Factor * compute_L1(model) + self.l2Factor * compute_L2(
            model)


def compute_L1(model):
    allWeights = torch.cat([x.view(-1) for x in model.parameters()])
    loss = torch.norm(allWeights, 1)
    return loss


def compute_L2(model):
    allWeights = torch.cat([x.view(-1) for x in model.parameters()])
    loss = torch.norm(allWeights, 2)
    return loss
