import os, sys
from os.path import join, isdir, isfile

import math
import pandas as pd
import numpy as np
import cv2
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

DEBUG = 0
os.makedirs("outputs", exist_ok=True)

PASCAL_VOC_CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]
def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    list_with_all_boxes = []

    im_width = int(root.find('size/width').text)
    im_height = int(root.find('size/height').text)
    for boxes in root.iter('object'):

        filename = root.find('filename').text

        ymin, xmin, ymax, xmax = None, None, None, None
        cls_name = str(boxes.find("name").text)
        ymin = int(boxes.find("bndbox/ymin").text)
        xmin = int(boxes.find("bndbox/xmin").text)
        ymax = int(boxes.find("bndbox/ymax").text)
        xmax = int(boxes.find("bndbox/xmax").text)
        cls_index = PASCAL_VOC_CLASSES.index(cls_name)
        x_center, y_center = (xmin+xmax)/2, (ymin+ymax)/2
        width, height = xmax-xmin, ymax-ymin
        # x_center, y_center = x_center / im_width, y_center / im_height
        # width, height = width / im_width, height / im_height
        list_with_single_boxes = [cls_index, x_center, y_center, width, height]
        list_with_all_boxes.append(list_with_single_boxes)

    return list_with_all_boxes


class PascalVoc(Dataset):
    def __init__(self, data_folder, split='train', S=3, B=2, C=20, transform=None):
        if split == "train":
            im_folder = join(data_folder, "VOC2012_train_val", "JPEGImages")
            imageset_file = join(data_folder, "VOC2012_train_val", "ImageSets", "Main", "train.txt")
            label_folder = join(data_folder, "VOC2012_train_val", "Annotations")
        elif split == "val":    
            im_folder = join(data_folder, "VOC2012_train_val", "JPEGImages")
            imageset_file = join(data_folder, "VOC2012_train_val", "ImageSets", "Main", "val.txt")
            label_folder = join(data_folder, "VOC2012_train_val", "Annotations")
        else:
            im_folder = join(data_folder, "VOC2012_test", "JPEGImages")
            imageset_file = join(data_folder, "VOC2012_test", "ImageSets", "Main", "test.txt")
            label_folder = join(data_folder, "VOC2012_train_val", "Annotations")

        with open(imageset_file, "r") as handle:
            lines = handle.readlines()
            lines = [l.rstrip() for l in lines]
        self.img_paths = []
        self.label_paths = []
        for l in lines:
            im_path = join(im_folder, l + ".jpg")
            label_path = join(label_folder, l + ".xml")
            self.img_paths.append(im_path)
            self.label_paths.append(label_path)
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform
        print(f"number of images: {self.__len__()}")

    def __len__(self):
        return 10
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        image = cv2.imread(image_path)
        boxes = read_content(label_path)
        boxes = np.array(boxes)
        if self.transform is not None:
            image, boxes = self.transform([image, boxes])
        label = np.zeros([self.S, self.S, self.C + 5])

        for box in boxes:
            cls, cx, cy, w, h = box
            cls = int(cls)
            j, i = math.floor(cx*self.S), math.floor(cy*self.S)
            offset_cx, offset_cy = cx*self.S - j, cy*self.S - i
            
            scaled_w, scaled_h = w*self.S, h*self.S

            label[i, j, cls] = 1
            label[i, j, self.C] = 1
            label[i, j, self.C + 1] = offset_cx
            label[i, j, self.C + 2] = offset_cy
            label[i, j, self.C + 3] = scaled_w
            label[i, j, self.C + 4] = scaled_h
        image = ToTensor()(image)
        label = torch.from_numpy(label)
        return image, label

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:

    """

    def __init__(self, max_size=448):
        self.max_size = max_size

    def __call__(self, sample):
        image, boxes = sample
        height, width, channels = image.shape
        if height > width:
            ratio = self.max_size / height
        else:
            ratio = self.max_size / width

        new_h, new_w = int(height*ratio), int(width*ratio)
        img = cv2.resize(image, (new_w, new_h))
        boxes[:, 1] = boxes[:, 1] * ratio / new_w
        boxes[:, 2] = boxes[:, 2] * ratio / new_h
        boxes[:, 3] = boxes[:, 3] * ratio / new_w
        boxes[:, 4] = boxes[:, 4] * ratio / new_h
        new_img = np.zeros([self.max_size, self.max_size, 3], dtype=image.dtype)
        new_img[:new_h, :new_w, :] = img
        return new_img, boxes

def build_dataloader(args, split="train"):
    if split == "train":
        transform_fn = transforms.Compose([
            Rescale(max_size=448),
            
        ])
        shuffle = True
    else:
        transform_fn = transforms.Compose([
            Rescale(max_size=448),
        ])
        shuffle = False
    dataset = PascalVoc(data_folder=args.data_folder, split=split, S=args.S, B=args.B, C=args.C, transform=transform_fn)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    return dataloader

if __name__ == '__main__':
    DATA_DIR = "../../Datasets/PASCAL_VOC"
    train_transform  = transforms.Compose([
        Rescale(max_size=448),
    ])
    train_dataset = PascalVoc(data_folder=DATA_DIR, split='train', S=7, B=2, C=20, transform=train_transform)
    num_sample = train_dataset.__len__()
    for i in range(num_sample):
        item = train_dataset.__getitem__(i)
        image, label = item
        print(i, image.shape, label.shape)
        # break
    # raw_image = cv2.imread("../../Datasets/PASCAL_VOC/VOC2012_train_val/JPEGImages/2008_000365.jpg")
    # h, w, _ = raw_image.shape
    # # print("label", label)
    # S = 7
    # for ii in range(S):
    #     for jj in range(S):
    #         if label[ii, jj, 20] != 1:
    #             continue
    #         offset_x, offset_y, scale_w, scale_h = label[ii, jj, 21], label[ii, jj, 22], label[ii, jj, 23], label[ii, jj, 24]
    #         cx = (offset_x + jj)/S
    #         cy = (offset_y + ii)/S
    #         ori_w, ori_h = scale_w / S, scale_h/S
    
    #         print("offset_x, offset_y, scale_w, scale_h", offset_x, offset_y, scale_w, scale_h)
    #         print("cx, cy, ori_w, ori_h", cx, cy, ori_w, ori_h)
    #         start_point = (cx - ori_w/2)*w, (cy - ori_h/2)*h
    #         end_point = (cx + ori_w/2)*w, (cy + ori_h/2)*h
    #         start_point = (int(start_point[0]), int(start_point[1]))
    #         end_point = (int(end_point[0]), int(end_point[1]))
    #         color = (0, 255, 0)
    #         thickness = 2

    #         raw_image = cv2.rectangle(raw_image, start_point, end_point, color, thickness) 
    #         raw_image = cv2.circle(raw_image, (int(cx), int(cy)), 2, color, thickness)
    # cv2.imwrite("outputs/debug.jpg", raw_image)