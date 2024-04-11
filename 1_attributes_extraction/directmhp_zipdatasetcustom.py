import os
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from utils.augmentations import letterbox


class _ImageZipDataset(torch.utils.data.Dataset):
    def __init__(self, zip_file, samples, imgsize, stride, auto):
        self.zip_file = zip_file
        self.samples = samples
        self.imgsize = imgsize
        self.stride = stride
        self.auto = auto

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        with self.zip_file.open(path) as f:
            img0 = np.asarray(Image.open(f))  # RGB
            # Padded resize
            img = letterbox(img0, self.imgsize, stride=self.stride, auto=self.auto)[0]

            # Convert
            img = img.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

        return path, img, img0, None

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        return fmt_str


class ImageZipDatasetMHP(torch.utils.data.Dataset):
    def __init__(self, zip_path, img_size=640, stride=32, auto=True):
        if not os.path.exists(zip_path):
            raise RuntimeError("%s does not exist" % zip_path)

        self.zip_path = zip_path
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.zipfile = ZipFile(self.zip_path, "r")
        files = self.zipfile.namelist()
        self.samples = []
        for file in files:
            ext = Path(file).suffix.lower()
            if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
                continue
            self.samples.append((file, 0))

    @contextmanager
    def dataset(self):
        res = _ImageZipDataset(
            zip_file=self.zipfile,
            samples=self.samples,
            imgsize=self.img_size,
            stride=self.stride,
            auto=self.auto,
        )
        yield res

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += f"    Number of datapoints: {self.__len__()}\n"
        fmt_str += f"    Zip Location: {self.zip_path}\n"
        return fmt_str
