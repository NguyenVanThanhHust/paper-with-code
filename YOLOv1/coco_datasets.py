from os.path import join, isdir, isfile
from pycocotools.coco import COCO
import numpy as np
from PIL import Image

import cv2
import math
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

class CocoDetectionDataset(Dataset):
    def __init__(self, image_folder, annFile, S=7, B=1, C=80, transform=None) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.coco = COCO(annFile)
        self.img_ids = self.coco.getImgIds()
        self.transform = transform
        coco_categories = sorted(self.coco.getCatIds())
        self.coco_id_to_contiguous_id = {coco_id: i + 1 for i, coco_id in enumerate(coco_categories)}
        self.contiguous_id_to_coco_id = {v: k for k, v in self.coco_id_to_contiguous_id.items()}
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        try:
            img_id = self.img_ids[index]
            img_name = self.coco.imgs[img_id]['file_name']
            img_path = join(self.image_folder, img_name)
            # image = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
            image = cv2.imread(img_path)
            im_h, im_w, im_c = image.shape
            annIds =self. coco.getAnnIds(imgIds=self.coco.imgs[img_id]['id'],iscrowd=None)
            anns = self.coco.loadAnns(annIds)
            label = np.zeros([self.S, self.S, self.C + 5])

            boxes = [] 
            for ann in anns:
                bbox = ann['bbox']
                x, y, w, h = bbox
                category_id = ann['category_id']
                cx = x + w / 2
                cy = y + h / 2
                cx = cx / im_w
                cy = cy / im_h
                w = w / im_w 
                h = h / im_h
                # cls = self.contiguous_id_to_coco_id[category_id]
                cls = self.coco_id_to_contiguous_id[category_id]
                box = [cls, cx, cy, h, w]
                boxes.append(box)
            boxes = np.array(boxes)
            if self.transform is not None:
                image, boxes = self.transform([image, boxes])

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
        except Exception as e:
            import os, sys
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(index, img_name, label)
            import pdb; pdb.set_trace()
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:

    """

    def __init__(self, max_size=448):
        self.max_size = max_size

    def __call__(self, sample):
        image, boxes = sample
        height, width, channels = image.shape
        max_hw = max(height, width)
        if max_hw > self.max_size:
            ratio = self.max_size / max_hw
        else:
            ratio = max_hw / self.max_size
        new_h, new_w = int(height*ratio), int(width*ratio)
        img = cv2.resize(image, (new_w, new_h))
        if len(boxes) > 0:
            boxes[:, 1] = boxes[:, 1] * ratio # / new_w
            boxes[:, 2] = boxes[:, 2] * ratio # / new_h
            boxes[:, 3] = boxes[:, 3] * ratio # / new_w
            boxes[:, 4] = boxes[:, 4] * ratio # / new_h
        new_img = np.zeros([self.max_size, self.max_size, 3], dtype=image.dtype)
        new_img[:new_h, :new_w, :] = img
        return new_img, boxes
    
def build_dataloader(args, split="train"):
    if split == "train":
        transform_fn = transforms.Compose([
            Rescale(max_size=448),
        ])
        shuffle = True
        coco_dataset = CocoDetectionDataset(image_folder="../../input/COCO/images/train2017", 
                                            annFile="../../input/COCO/annotations/instances_train2017.json",
                                            transform=transform_fn)
    else:
        transform_fn = transforms.Compose([
            Rescale(max_size=448),
        ])
        shuffle = False
        coco_dataset = CocoDetectionDataset(image_folder="../../input/COCO/images/val2017", 
                                            annFile="../../input/COCO/annotations/instances_val2017.json",
                                            transform=transform_fn)
    dataloader = DataLoader(dataset=coco_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers)
    return dataloader


if __name__ == '__main__':
    val_transform = transforms.Compose([
        Rescale(max_size=448),
    ])
    coco_dataset = CocoDetectionDataset(image_folder="../../input/COCO/images/val2017", 
                                        annFile="../../input/COCO/annotations/instances_val2017.json",
                                        transform=val_transform)
    # for i in range(len(coco_dataset)):
    #     im, tgt = coco_dataset.__getitem__(i)
    #     print(i, im.shape, tgt.shape) 
    print(len(coco_dataset))
    coco_dataloader = DataLoader(coco_dataset, 2, False)
    for idx, (ims, tgts) in enumerate(coco_dataloader):
        print(idx, ims.shape, tgts.shape)