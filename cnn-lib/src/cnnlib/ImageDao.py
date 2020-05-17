from zipfile import ZipFile
from cnnlib import ImageUtils
from io import BytesIO


class ZipFileImagePersister:

    def __init__(self, zip, out_folder_name, format="JPEG"):
        self.out_folder = out_folder_name
        self.format = format
        self.zip = zip

    def __call__(self, images, names=None):

        if names is not None:
            assert len(images) == len(names)
        else:
            names = [f"Image_{i}.{self.format}" for i in range(0, len(images))]

        for i, img in enumerate(images):
            with BytesIO() as buf:
                img.save(buf, format=self.format)
                self.zip.writestr(f"{self.out_folder}/{names[i]}", buf.getvalue())
        return names


class ImagesPlotter:

    def __init__(self, cols=10, fig_size=(15, 15)):
        self.cols = cols
        self.fig_size = fig_size

    def __call__(self, images, names):
        ImageUtils.show_images(images, titles=names, cols=self.cols, figSize=self.fig_size)
        return names
