import torch
import csv
from os import listdir
from os.path import isfile, join
from PIL import Image
import gc
from tqdm import tqdm_notebook as tqdm
from zipfile import ZipFile


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


def loadTsvAsDict(file):
    with open(file, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        d = dict([(row[0], row[1]) for row in reader])
    return d


def loadFileToArray(file):
    with open(file, 'r') as f:
        x = f.read().splitlines()
    return x


def load_images_to_dict(folder, extn):
    imgFiles = [join(folder, f) for f in listdir(folder)]
    imagesDict = {}
    for imgPath in imgFiles:
        if imgPath.endswith(extn):
            with open(imgPath, 'rb') as f:
                x = Image.open(f)
                x = x.convert('RGB')
                imagesDict[imgPath] = x
    return imagesDict


def load_image(file, channel):
    with open(file, 'rb') as f:
        x = Image.open(f)
        x = x.convert(channel)
    return x


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def unzip(input_zip, output_dir):
    zf = ZipFile(input_zip)
    uncompress_size = sum((file.file_size for file in zf.infolist()))
    print('uncompressed_size', uncompress_size / 1e6)

    pbar = tqdm(zf.infolist(), ncols=1000)
    # Loop through all files attempting to decompress each individually
    extracted_size = 0
    for idx, file in enumerate(pbar):
        extracted_size += file.file_size
        try:
            zf.extract(file, output_dir)
        except:
            continue

    zf.close()
