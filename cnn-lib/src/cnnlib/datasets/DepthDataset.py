import torch.utils.data as data
import os
from os.path import join
from PIL import Image
import re
from cnnlib.ImageUtils import load_image


class DepthDataset(data.Dataset):

    def __init__(self, input_folder, bg_transform=None, fg_bg_transform=None, fg_bg_mask_transform=None,
                 fg_bg_depth_transform=None, reshape=None):
        self.input_folder = input_folder
        self.fg_bg_files = os.listdir(join(input_folder, "fg_bg"))
        self.bg_files = os.listdir(join(input_folder, "bg_images"))
        self.fg_bg_mask_files = os.listdir(join(input_folder, "fg_bg_mask"))
        self.fg_bg_depth_files = os.listdir(join(input_folder, "fg_bg_depth"))

        self.bg_transform = bg_transform
        self.fg_bg_transform = fg_bg_transform
        self.fg_bg_mask_transform = fg_bg_mask_transform
        self.fg_bg_depth_transform = fg_bg_depth_transform
        self.reshape = reshape

    def __getitem__(self, index):

        fg_bg = load_image(join(self.input_folder, "fg_bg"), self.fg_bg_files[index], 'RGB')
        bg = load_image(join(self.input_folder, "bg_images"),
                        self.bg_files[self.__get_bg_index__(self.fg_bg_files[index])], 'RGB')
        fg_bg_mask = load_image(join(self.input_folder, "fg_bg_mask"), self.fg_bg_files[index], 'L')
        fg_bg_depth = load_image(join(self.input_folder, "fg_bg_depth"), self.fg_bg_files[index], 'L')
        name = self.fg_bg_files[index]

        if self.reshape:
            fg_bg = fg_bg.resize(self.reshape, Image.BILINEAR)
            bg = bg.resize(self.reshape, Image.BILINEAR)
            fg_bg_mask = fg_bg_mask.resize(self.reshape, Image.BILINEAR)
            fg_bg_depth = fg_bg_depth.resize(self.reshape, Image.BILINEAR)

        if self.bg_transform:
            bg = self.bg_transform(bg)

        if self.fg_bg_transform:
            fg_bg = self.fg_bg_transform(fg_bg)

        if self.fg_bg_mask_transform:
            fg_bg_mask = self.fg_bg_mask_transform(fg_bg_mask)

        if self.fg_bg_depth_transform:
            fg_bg_depth = self.fg_bg_depth_transform(fg_bg_depth)

        return dict({'name': name, 'fg_bg': fg_bg, 'bg': bg, 'fg_bg_mask': fg_bg_mask, 'fg_bg_depth': fg_bg_depth})

    def __len__(self):
        return len(self.fg_bg_files)

    def __get_bg_index__(self, file_name):
        return int(re.search("Image_([0-9]+).*", file_name).group(1))
