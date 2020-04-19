from torchvision.datasets.vision import VisionDataset
from torchvision.datasets import ImageFolder
from cnnlib import Utility
from os import listdir
from os.path import isfile, join


class TinyImageNet(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.train = train
        self.__load_classes__(root)
        if (train):
            self.train_folder = join(root, "train")
            self.__prepare_train__(root)
        else:
            self.test_images_folder = join(root, "val/images")
            self.test_annotations_folder = join(root, "val")
            self.__prepare_test__(root)

    def __load_classes__(self, root):
        self.idx_class = Utility.loadFileToArray(root + "/wnids.txt")
        self.class_idx = dict((c, i) for i, c in enumerate(self.idx_class))

    def __prepare_train__(self, root):
        self.trainImageFolder = ImageFolder(self.train_folder)
        cidx = self.trainImageFolder.class_to_idx
        self.trainImageFolder_idx_to_class = dict((cidx[k], k) for k in cidx)
        self.data_size = len(self.trainImageFolder)

    def __prepare_test__(self, root):
        self.test_images_dict = Utility.load_images_to_dict(self.test_images_folder, "JPEG")
        self.test_images_files = list(self.test_images_dict.keys())

        self.test_truth_labels = Utility.loadTsvAsDict(join(self.test_annotations_folder, "val_annotations.txt"))
        self.test_truth_labels = dict(
            (join(self.test_images_folder, f), self.test_truth_labels[f]) for f in self.test_truth_labels)
        self.data_size = len(self.test_images_files)

    def __getitem__(self, index):

        if (self.train):
            x, class_y = self.__train_get_item__(index)
        else:
            x, class_y = self.__test_get_item__(index)

        if self.transform:
            x = self.transform(x)

        y = self.class_idx[class_y]
        return x, y

    def __train_get_item__(self, index):
        x, y = self.trainImageFolder.__getitem__(index)
        # Convert y to class
        class_y = self.trainImageFolder_idx_to_class[y]
        return x, class_y

    def __test_get_item__(self, index):
        imgFile = self.test_images_files[index]
        x = self.test_images_dict[imgFile]
        class_y = self.test_truth_labels[imgFile]
        return x, class_y

    def __len__(self):
        return self.data_size
