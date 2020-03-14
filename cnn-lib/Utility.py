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


def unnormalize(tensor, mean, sig):
    """
    Unnormalize the tensor
    """
    return tensor * sig + mean
