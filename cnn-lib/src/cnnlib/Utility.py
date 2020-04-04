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

def pickRandomElements(data, count):
    randIndex = randInt(0, len(data), count)
    if (count >= len(data)):
        randIndex = [i for i in range(0, len(data))]

    return randIndex

def unnormalize(images, mean, sig):
    """
    Unnormalize the tensor
    """
    copy = images.clone().detach()

    for img in copy:
        for t, m, s in zip(img, mean, sig):
            t.mul_(s).add_(m)
    return copy

def toImages(data, muSigPair):
    if muSigPair:
        data = unnormalize(data, muSigPair[0], muSigPair[1])
    return data.permute(0, 2, 3, 1)