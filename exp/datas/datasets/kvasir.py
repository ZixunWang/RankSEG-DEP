from glob import glob

import cv2
import numpy as np

from .base_dataset import BaseDataset


class Kvasir(BaseDataset):
    def __init__(self,
                 train=True,
                 data_dir="/Users/whoami/datasets",
                 in_channels=3,
                 num_classes=2,
                 scale_min=0.5,
                 scale_max=2.0,
                 crop_size=[512, 512],
                 ignore_index=2,
                 reduce_zero_label=False,
                 image_prefix="/images/",
                 image_suffix=".jpg",
                 label_prefix="/masks/",
                 label_suffix=".jpg",
                 **kwargs):
        super().__init__(train=train,
                         data_dir=data_dir,
                         in_channels=in_channels,
                         num_classes=num_classes,
                         scale_min=scale_min,
                         scale_max=scale_max,
                         crop_size=crop_size,
                         ignore_index=ignore_index,
                         reduce_zero_label=reduce_zero_label,
                         image_prefix=image_prefix,
                         image_suffix=image_suffix,
                         label_prefix=label_prefix,
                         label_suffix=label_suffix)

        self.image_list = self.get_image_list()
        self.transform = self.get_transform()
        self.class_names = self.get_class_names()
        self.color_map = self.get_color_map()


    def __getitem__(self, index):
        image_file = self.image_list[index]
        label_file = self.get_label_file(image_file)

        image_bgr = cv2.imread(image_file, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = np.float32(image_rgb)

        label = cv2.imread(label_file, cv2.IMREAD_GRAYSCALE)
        label = np.uint8(label > 127)

        image, label = self.transform(image, label)

        assert image.shape[1:3] == label.shape[0:2]

        return image, label, image_file


    def get_image_list(self):
        image_list = glob(f"{self.data_dir}/Kvasir-SEG/images/*.jpg")
        image_list.sort()

        assert len(image_list) == 1000, f"len(image_list) == {len(image_list)} != 1000"

        split = int(len(image_list) * 0.8)
        train_image_list = image_list[:split]
        val_image_list = image_list[split:]

        assert len(train_image_list) == 800
        assert len(val_image_list) == 200

        if self.train:
            return train_image_list
        else:
            return val_image_list


    def get_class_names(self):
        class_names = ["Background", "Polyp"]

        return class_names


    def get_color_map(self):
        color_map = [[0, 0, 0], [255, 255, 255]]

        return color_map
