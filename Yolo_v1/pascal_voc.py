import marimo

__generated_with = "0.1.88"
app = marimo.App()


@app.cell
def __():
    import os
    from os.path import join, isdir, isfile

    import math
    import pandas as pd
    import numpy as np
    import cv2
    import torch

    from torch.utils.data import Dataset, DataLoader
    from torchvision import transforms, utils

    import matplotlib.pyplot as plt
    return (
        DataLoader,
        Dataset,
        cv2,
        isdir,
        isfile,
        join,
        math,
        np,
        os,
        pd,
        plt,
        torch,
        transforms,
        utils,
    )


@app.cell
def __(isdir):
    DATA_DIR = "../../Datasets/PASCAL_VOC"
    isdir(DATA_DIR)
    return DATA_DIR,


@app.cell
def __(os):
    DEBUG = 0
    os.makedirs("outputs", exist_ok=True)
    return DEBUG,


@app.cell
def __(DATA_DIR, join, os):
    im_folder = join(DATA_DIR, 'images')
    im_names = next(os.walk(im_folder))[2]
    print(im_names[:3])
    return im_folder, im_names


@app.cell
def __(DATA_DIR, join, pd):
    train_df = pd.read_csv(join(DATA_DIR, 'train.csv'))
    print(train_df.head())
    print(train_df.shape)
    return train_df,


@app.cell
def __(DATA_DIR, join, pd):
    test_df = pd.read_csv(join(DATA_DIR, 'test.csv'))
    print(test_df.head())
    print(test_df.shape)
    return test_df,


@app.cell
def __(Dataset, cv2, join, math, np, pd):
    class PascalVoc(Dataset):
        def __init__(self, data_folder, split='train', S=3, B=2, C=20, transform=None):
            self.image_folder = join(data_folder, "images")
            self.df = pd.read_csv(join(data_folder, split+".csv"))
            self.label_folder = join(data_folder, "labels")
            self.S = S
            self.B = B
            self.C = C
            self.transform = transform

        def __len__(self):
            return self.df.shape[0]

        def __getitem__(self, idx):
            image_name = self.df.iloc[idx].values[0]
            label_name = self.df.iloc[idx].values[1]
            image_path = join(self.image_folder, image_name)
            label_path = join(self.label_folder, label_name)
            image = cv2.imread(image_path)
            boxes = []
            with open(label_path, 'r') as handle:
                lines = handle.readlines()
                lines = [l.rstrip() for l in lines]
            for line in lines:
                info = line.split(' ')
                cls, x_center, y_center, width, height = float(info[0]), float(info[1]), float(info[2]), float(info[3]), float(info[4])
                boxes.append([cls, x_center, y_center, width, height])
            boxes = np.array(boxes)
            if self.transform is not None:
                image, boxes = self.transform([image, boxes])
            label = np.zeros([self.S, self.S, self.C + 5])

            for box in boxes:
                cls, cx, cy, w, h = box
                cls = int(cls)
                j, i = math.floor(cx*self.S), int(cy*self.S)
                offset_cx, offset_cy = cx*self.S - j, cy*self.S - i
                
                scaled_w, scaled_h = w*self.S, h*self.S

                label[i, j, cls] = 1
                label[i, j, self.C] = 1
                label[i, j, self.C + 1] = offset_cx
                label[i, j, self.C + 2] = offset_cy
                label[i, j, self.C + 3] = scaled_w
                label[i, j, self.C + 4] = scaled_h

            return image, label
    return PascalVoc,


@app.cell
def __(cv2, np):
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
                new_h = self.max_size
                new_w = int(ratio * width)
            else:
                ratio = self.max_size / width
                new_w = self.max_size
                new_h = int(ratio * height)

            new_h, new_w = int(new_h), int(new_w)

            img = cv2.resize(image, (new_w, new_h))
            boxes[:, 1] = boxes[:, 1] * new_w / width
            boxes[:, 2] = boxes[:, 2] * new_h / height
            boxes[:, 3] = boxes[:, 3] * new_w / width
            boxes[:, 4] = boxes[:, 4] * new_h / height

            new_img = np.zeros([self.max_size, self.max_size, 3], dtype=image.dtype)
            new_img[:new_h, :new_w, :] = img
            return new_img, boxes
    return Rescale,


@app.cell
def __(DATA_DIR, DEBUG, PascalVoc, Rescale, transforms):
    if DEBUG==1:
        train_transform  = transforms.Compose([
            Rescale(max_size=448),
        ])
        train_dataset = PascalVoc(data_folder=DATA_DIR, split='train', S=3, B=2, C=20, transform=None)
        
        for i in range(1):
            item = train_dataset.__getitem__(i)
            image, label = item
            print(image.shape, label.shape)
    return i, image, item, label, train_dataset, train_transform


@app.cell
def __(DEBUG, cv2):
    if DEBUG==1:
        raw_image = cv2.imread("../../Datasets/PASCAL_VOC/images/000007.jpg")
        h, w, _ = raw_image.shape
    return h, raw_image, w


@app.cell
def __(DEBUG, cv2, h, label, plt, raw_image, w):
    if DEBUG==1:
        for ii in range(3):
            for jj in range(3):
                if label[ii, jj, 20] != 1:
                    continue
                offset_x, offset_y, scale_w, scale_h = label[ii, jj, 21], label[ii, jj, 22], label[ii, jj, 23], label[ii, jj, 24]
                cx = (offset_x + jj)/3
                cy = (offset_y + ii)/3
                ori_w, ori_h = scale_w / 3, scale_h/3
        
                print("offset_x, offset_y, scale_w, scale_h", offset_x, offset_y, scale_w, scale_h)
                print("cx, cy, ori_w, ori_h", cx, cy, ori_w, ori_h)
        start_point = (cx - ori_w/2)*w, (cy - ori_h/2)*h
        end_point = (cx + ori_w/2)*w, (cy + ori_h/2)*h
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point = (int(end_point[0]), int(end_point[1]))
        color = (0, 255, 0)
        thickness = 2

        new_image = cv2.rectangle(raw_image, start_point, end_point, color, thickness) 
        new_image = cv2.circle(raw_image, (int(cx), int(cy)), 2, color, thickness)
        cv2.imwrite("outputs/debug.jpg", new_image)
        plt.imshow(new_image)
    return (
        color,
        cx,
        cy,
        end_point,
        ii,
        jj,
        new_image,
        offset_x,
        offset_y,
        ori_h,
        ori_w,
        scale_h,
        scale_w,
        start_point,
        thickness,
    )


@app.cell
def __(DATA_DIR, DEBUG, color, cv2, join, pd, thickness):
    if DEBUG==1:
        import copy
        data_folder = DATA_DIR
        image_folder = join(data_folder, "images")
        df = pd.read_csv(join(data_folder, "train.csv"))
        label_folder = join(data_folder, "labels")
        for idx in range(1):
            image_name = df.iloc[idx].values[0]
            label_name = df.iloc[idx].values[1]
            image_path = join(image_folder, image_name)
            label_path = join(label_folder, label_name)
            draw_image = cv2.imread(image_path)
            image_shape = draw_image.shape
            boxes = []
            with open(label_path, 'r') as handle:
                lines = handle.readlines()
                lines = [l.rstrip() for l in lines]
            for line in lines:
                info = line.split(' ')
                cls, x_center, y_center, width, height = float(info[0]), float(info[1]), float(info[2]), float(info[3]), float(info[4])
                print(x_center, y_center, width, height)
                tl = int((x_center - width/2)*image_shape[1]), int((y_center - height/2)*image_shape[0])
                br = int((x_center + width/2)*image_shape[1]), int((y_center + height/2)*image_shape[0])
                draw_image = cv2.rectangle(draw_image, tl, br, color, thickness)
            cv2.imwrite("outputs/{}.jpg".format(idx), draw_image)
    return (
        boxes,
        br,
        cls,
        copy,
        data_folder,
        df,
        draw_image,
        handle,
        height,
        idx,
        image_folder,
        image_name,
        image_path,
        image_shape,
        info,
        label_folder,
        label_name,
        label_path,
        line,
        lines,
        tl,
        width,
        x_center,
        y_center,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
