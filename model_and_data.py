import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
import numpy as np
import os
from torchvision import datasets, transforms
from typing import List
from PIL import Image
class FCN_ResNet(nn.Module):
    def __init__(self, num_classes):
        super(FCN_ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x



# ---------------- VOC2012 语义分割（本地已下载目录）----------------
NUM_CLASSES = 21  # VOC2012（含背景）


class VOC2012SegLocal(Dataset):
    """从已下载的 VOC2012 目录读取分割数据。
    期望目录结构：
      voc2012_dir/
        JPEGImages/
        SegmentationClass/
        ImageSets/Segmentation/train.txt, val.txt, trainval.txt
    """

    def __init__(self, voc2012_dir: str = "./VOC2012", split: str = "train", resize_size: int = 512):
        self.voc2012_dir = voc2012_dir
        self.resize_size = resize_size
        self.ids_file = os.path.join(voc2012_dir, "ImageSets", "Segmentation", f"{split}.txt")
        if not os.path.exists(self.ids_file):
            raise FileNotFoundError(f"未找到划分列表: {self.ids_file}")
        with open(self.ids_file, "r") as f:
            self.ids: List[str] = [line.strip() for line in f if line.strip()]

        self.jpg_dir = os.path.join(voc2012_dir, "JPEGImages")
        self.mask_dir = os.path.join(voc2012_dir, "SegmentationClass")

        # 与 ImageNet 预训练一致的归一化
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = os.path.join(self.jpg_dir, f"{image_id}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{image_id}.png")

        image = Image.open(img_path).convert("RGB")
        target = Image.open(mask_path)

        # 统一固定大小
        target_size = (self.resize_size, self.resize_size)
        image = TF.resize(image, target_size, interpolation=TF.InterpolationMode.BILINEAR)
        target = TF.resize(target, target_size, interpolation=TF.InterpolationMode.NEAREST)

        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=self.mean, std=self.std)

        target = np.array(target, dtype=np.uint8)
        target = torch.as_tensor(target, dtype=torch.long)

        return image, target


def get_voc_dataloader_local(
    voc2012_dir: str,
    split: str = "train",
    resize_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
    pin_memory: bool = True,
):
    ds = VOC2012SegLocal(voc2012_dir=voc2012_dir, split=split, resize_size=resize_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

