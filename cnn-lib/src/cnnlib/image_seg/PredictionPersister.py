from zipfile import ZipFile
from cnnlib.ImageDao import ZipFileImagePersister
from torchvision import transforms
from cnnlib import Utility
import torch
from cnnlib import ImageUtils


class ZipPredictionPersister:

    def __init__(self, zip_file_name):
        self.zip_file_name = zip_file_name
        self.pil_transform = transforms.ToPILImage()
        self.out_folder_pattern = "epoch-{}/fg_bg_{}"

    def __call__(self, data, pred, epoch, type):
        pred = pred.to(torch.device("cpu"))
        names = data['name']
        zip = ZipFile(self.zip_file_name, 'a')
        persister = ZipFileImagePersister(zip, out_folder_name=self.out_folder_pattern.format(epoch, type))
        pred_images = [self.pil_transform(t) for t in pred]
        persister(pred_images, names)
        zip.close()


if __name__ == '__main__':

    persister = ZipPredictionPersister("data.zip")
    trans = transforms.Compose([transforms.ToTensor()])

    tensor = trans(Utility.load_image("data/tiny_data/fg_bg_mask/Image_0_0_4.JPEG", 'L'))
    print(tensor.shape)

    pil = transforms.ToPILImage()
    img = pil(tensor)

    print(img.size)
    ImageUtils.show_images([img])