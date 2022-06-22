from xml import etree

import cv2
from torch.utils.data import Dataset
import os
import torch
import json
from PIL import Image


class Cervical_cancer_DataSet(Dataset):
    """ 读取数据集"""

    def __init__(self, data_root, transforms=None, txt_name: str = "train.txt"):
        self.img_root = os.path.join(data_root, 'images')
        self.annotations = os.path.join(data_root, 'labels')

        # read train.txt or val.txt file
        txt_path = os.path.join(data_root, txt_name)
        assert os.path.exists(txt_path), "not found {} file.".format(txt_name)

        with open(txt_path) as f:
            self.txt_list = [os.path.join(self.annotations, line.strip() + '.txt')
                             for line in f.readlines() if len(line.strip()) > 0]
        assert len(self.txt_list) > 0, "in '{}' file does not find any information".format(txt_path)

        # read class_indict
        json_file = './classes.json'
        assert os.path.exists(json_file), '{} file not exist'.format(json_file)
        with open(json_file, 'r') as f:
            self.class_dict = json.load(f)

        self.transforms = transforms

    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, idx):
        # read labels txt file
        txt_path = self.txt_list[idx]
        with open(txt_path) as f:
            data = f.readlines()[0].split(' ')

        image_path = os.path.join(self.img_root, data[0]) + '.jpg'
        # print(image_path)
        image = Image.open(image_path).convert("RGB")

        boxes = []
        labels = []
        iscrowd = []
        x_min = float(data[1])
        y_min = float(data[2])
        x_max = x_min + float(data[3])
        y_max = y_min + float(data[4])
        # print(x_min, y_min, x_max, y_max)

        # 检查数据，过滤标注信息中可能w或h为0的情况，这样的数据会导致计算回归loss为Nan
        if x_max < x_min or y_max < y_min:
            print("Warning: in '{}' txt, there are some bbox w/h <=0".format(txt_path))
            x_min = y_min = x_max = y_max = 0

        boxes.append([x_min, y_min, x_max, y_max])

        labels.append(int(data[-1]))

        iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        # print(image, target)
        return image, target

    def get_height_and_width(self, idx):
        # read labels txt file
        # txt_path = self.txt_list[idx]
        # with open(txt_path) as f:
        #     data = f.readlines()[0].split(' ')
        # image_path = os.path.join(self.img_root, data[0]) + '.jpg'
        # image = cv2.imread(image_path)

        data_height, data_width = 2028, 2428

        return data_height, data_width

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))

    def coco_index(self, idx):
        """
        该方法是专门为pycocotools统计标签信息准备，不对图像和标签作任何处理
        由于不用去读取图片，可大幅缩减统计时间

        Args:
            idx: 输入需要获取图像的索引
        """
        # read xml
        # read labels txt file
        txt_path = self.txt_list[idx]
        with open(txt_path) as f:
            data = f.readlines()[0].split(' ')
        # img_path = os.path.join(self.img_root, data["filename"])
        # image = Image.open(img_path)
        # if image.format != "JPEG":
        #     raise ValueError("Image format not JPEG")
        boxes = []
        labels = []
        iscrowd = []
        x_min = float(data[1])
        y_min = float(data[2])
        x_max = x_min + float(data[3])
        y_max = y_min + float(data[4])

        boxes.append([x_min, y_min, x_max, y_max])

        labels.append(int(data[-1]))

        iscrowd.append(0)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        return (float(data[4]), float(data[3])), target

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


# (<PIL.Image.Image image mode=RGB size=2428x2028 at 0x1FB08B50370>, {'boxes': tensor([[ 951.,  417., 1131.,  603.]]), 'labels': tensor([3]), 'image_id': tensor([0]), 'area': tensor([33480.]), 'iscrowd': tensor([0])})
# (<PIL.Image.Image image mode=RGB size=3642x3042 at 0x1EF6A86ADC0>, {'boxes': tensor([[ 754.,  600., 1279., 1024.]]), 'labels': tensor([3]), 'image_id': tensor([0]), 'area': tensor([222600.]), 'iscrowd': tensor([0])})
if __name__ == '__main__':
    data = Cervical_cancer_DataSet(data_root='H:/dataset/cut/')
    print(data.__getitem__(0))
