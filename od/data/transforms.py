import random

import cv2
import numpy
import numpy as np
import torch
from PIL import Image

from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    """将PIL图像转为Tensor"""

    def __call__(self, image, target):
        image = F.to_tensor(image)

        return image, target


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target


class RandomRotate90(object):
    """ 随机逆时针旋转factor个90度以及bboxes"""

    def __init__(self, prob=0.5, factor=1):
        self.prob = prob
        self.factor = factor

    def bbox_rot90(self, bbox_list, factor, width, height):
        """Rotates a bounding box by 90 degrees CCW (see np.rot90)

        Args:
            bbox (tuple): A bounding box tuple (x_min, y_min, x_max, y_max).
            factor (int): Number of CCW rotations. Must be in set {0, 1, 2, 3} See np.rot90.
            rows (int): Image rows.
            cols (int): Image cols.

        Returns:
            tuple: A bounding box tuple (x_min, y_min, x_max, y_max).

        """
        # global bbox
        if factor not in {0, 1, 2, 3}:
            raise ValueError("Parameter n must be in set {0, 1, 2, 3}")

        x_min, y_min, x_max, y_max = bbox_list[0][:]
        if factor == 1:
            bbox = y_min, width - x_max, y_max, width - x_min
        elif factor == 2:
            bbox = width - x_max, height - y_max, width - x_min, height - y_min
        elif factor == 3:
            bbox = height - y_max, x_min, height - y_min, x_max

        return bbox

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:][0], image.shape[-2:][1]
            bbox = target["boxes"]
            bbox_list = torch.Tensor(bbox).tolist()
            image = F.to_pil_image(image)
            img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
            image = np.rot90(img, self.factor)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = F.to_tensor(image)
            new_bbox = self.bbox_rot90(bbox_list, self.factor, width, height)
            target["boxes"] = torch.Tensor([new_bbox])

        return image, target


class RandomBrightnessContrast(object):

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, prob=0.5, brightness_by_max=True):
        self.brightness_limit = self.to_tuple(brightness_limit)
        self.contrast_limit = self.to_tuple(contrast_limit)
        self.prob = prob
        self.brightness_by_max = brightness_by_max
        self.MAX_VALUES_BY_DTYPE = {
            np.dtype("uint8"): 255,
            np.dtype("uint16"): 65535,
            np.dtype("uint32"): 4294967295,
            np.dtype("float32"): 1.0,
        }

    def to_tuple(self, param, low=None, bias=None):
        """Convert input argument to min-max tuple
        Args:
            param (scalar, tuple or list of 2+ elements): Input value.
                If value is scalar, return value would be (offset - value, offset + value).
                If value is tuple, return value would be value + offset (broadcasted).
            low:  Second element of tuple can be passed as optional argument
            bias: An offset factor added to each element
        """
        if low is not None and bias is not None:
            raise ValueError("Arguments low and bias are mutually exclusive")

        if param is None:
            return param

        if isinstance(param, (int, float)):
            if low is None:
                param = -param, +param
            else:
                param = (low, param) if low < param else (param, low)
        elif isinstance(param, (list, tuple)):
            param = tuple(param)
        else:
            raise ValueError("Argument param must be either scalar (int, float) or tuple")

        if bias is not None:
            return tuple(bias + x for x in param)

        return tuple(param)

    def brightness_contrast_adjust_uint(self, img, alpha, beta, beta_by_max):
        dtype = np.dtype("uint8")

        max_value = self.MAX_VALUES_BY_DTYPE[dtype]

        lut = np.arange(0, max_value + 1).astype("float32")

        if alpha != 1:
            lut *= alpha
        if beta != 0:
            if beta_by_max:
                lut += beta * max_value
            else:
                lut += beta * np.mean(img)

        lut = np.clip(lut, 0, max_value).astype(dtype)
        img = cv2.LUT(img, lut)
        return img

    def brightness_contrast_adjust_non_uint(self, img, alpha, beta, beta_by_max):
        dtype = img.dtype
        img = img.astype("float32")

        if alpha != 1:
            img *= alpha
        if beta != 0:
            if beta_by_max:
                max_value = self.MAX_VALUES_BY_DTYPE[dtype]
                img += beta * max_value
            else:
                img += beta * np.mean(img)
        return img

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.to_pil_image(image)
            img = cv2.cvtColor(numpy.asarray(image), cv2.COLOR_RGB2BGR)
            alpha = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])
            beta = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
            if img.dtype == np.uint8:
                image = self.brightness_contrast_adjust_uint(img, alpha, beta, self.brightness_by_max)
            else:
                image = self.brightness_contrast_adjust_non_uint(img, alpha, beta, self.brightness_by_max)
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image = F.to_tensor(image)

        return image, target
