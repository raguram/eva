from zipfile import ZipFile
from cnnlib.ImageDao import ZipFileImagePersister
from torchvision import transforms
from cnnlib import Utility
import torch


class ZipPredictionPersister:

    def __init__(self, zip_file_name, out_folder_pattern):
        self.zip_file_name = zip_file_name
        self.out_folder_pattern = out_folder_pattern
        self.pil_transform = transforms.ToPILImage()

    def __call__(self, data, pred, epoch):
        pred = pred.to(torch.device("cpu"))
        names = data['name']
        zip = ZipFile(self.zip_file_name, 'a')
        persister = ZipFileImagePersister(zip, out_folder_name=self.out_folder_pattern.format(epoch))
        pred_images = [self.pil_transform(t) for t in pred]
        persister(pred_images, names)
        zip.close()
