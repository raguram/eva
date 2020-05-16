from zipfile import ZipFile
from cnnlib import ImageUtils
from io import BytesIO


class ZipFileImagePersister:

    def __init__(self, zip_file_name, out_folder_name, format="JPEG"):
        self.zip_file_name = zip_file_name
        self.out_folder = out_folder_name
        self.format = format

    def __call__(self, images, names=None):

        if names is not None:
            assert len(images) == len(names)
        else:
            names = [f"Image_{i}.{self.format}" for i in range(0, len(images))]

        zip = ZipFile(self.zip_file_name, mode="a")

        for i, img in enumerate(images):
            with BytesIO() as buf:
                img.save(buf, format=self.format)
                zip.writestr(f"{self.out_folder}/{names[i]}", buf.getvalue())
        zip.close()


class ImagesPlotter:

    def __init__(self, cols=10, fig_size=(15, 15)):
        self.cols = cols
        self.fig_size = fig_size

    def __call__(self, images, names):
        ImageUtils.show_images(images, titles=names, cols=self.cols, figSize=self.fig_size)
