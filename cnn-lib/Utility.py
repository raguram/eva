import torch


def isCuda():
    return torch.cuda.is_available()


def randInt(min, max, size):
    return torch.LongTensor(size).random_(min, max)


def getDevice():
    return torch.device("cuda" if isCuda() else "cpu")


def setSeed(seed):
    torch.manual_seed(seed)
    if isCuda():
        torch.cuda.manual_seed(seed)


def randInt(min, max, size):
    return torch.LongTensor(size).random_(min, max)


def unnormalize(images, mean, sig):
    """
    Unnormalize the tensor
    """
    copy = images.clone().detach()

    for img in copy:
        for t, m, s in zip(img, mean, sig):
            t.mul_(s).add_(m)
    return copy
