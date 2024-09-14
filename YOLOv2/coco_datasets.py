from os.path import join, isdir, isfile
from pycocotools.coco import COCO
import numpy as np
from PIL import Image
import random

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transform import ToTensor
import torch
import torch.nn.functional as F
    
class CocoDetectionDataset(Dataset):
    def __init__(self, image_folder, annFile, multiscale=True, transform=None) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.coco = COCO(annFile)
        self.img_ids = self.coco.getImgIds()
        self.transform = transform
        self.img_sizes = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
        self.multiscale = multiscale
        self.batch_count = 0
        self.picked_size = 320

    def __len__(self):
        return len(self.img_ids)
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_name = self.coco.imgs[img_id]['file_name']
        img_path = join(self.image_folder, img_name)
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        annIds =self. coco.getAnnIds(imgIds=self.coco.imgs[img_id]['id'],iscrowd=None)
        anns = self.coco.loadAnns(annIds)
        bboxes = [] 
        for ann in anns:
            bbox = ann['bbox']
            x, y, w, h = bbox
            category_id = ann['category_id']
            bboxes.append([category_id, x, y, x+w, y+h])
        bboxes = np.array(bboxes)
        if self.transform is not None:
            try:
                img, bboxes = self.transform((img, bboxes))
            except:
                import pdb; pdb.set_trace()
        return img, bboxes
    
    @staticmethod
    def resize(img, size):
        return F.interpolate(img.unsqueeze(0), (size, size), mode='bilinear', align_corners=False).squeeze(0)
    
    def collate_fn(self, batch):
        self.batch_count += 1
        imgs, boxes = list(zip(*batch))
        if self.multiscale and self.batch_count%10==0:
            picked_size = random.choice(self.img_sizes)
            self.picked_size = picked_size
        
        # Resize all images to a fixed sizes
        imgs = [self.resize(img, self.picked_size) for img in imgs]
        imgs = torch.stack(imgs)

        # Convert boxes
        tgts = []
        for i, box in enumerate(boxes):
            box[:, 0] = i
            tgts.append(box)
        tgts = torch.cat(tgts, 0)
        return imgs, tgts
    
if __name__ == '__main__':
    val_transform = transforms.Compose([
        ToTensor()
    ])
    coco_dataset = CocoDetectionDataset(image_folder="/home/thanhnv154te/workspace/Datasets/COCO/val2017", 
                                        annFile="/home/thanhnv154te/workspace/Datasets/COCO/annotations_trainval2017/annotations/instances_val2017.json",
                                        multiscale=True,
                                        transform=val_transform,
                                        )
    coco_dataloader = DataLoader(coco_dataset, batch_size=2, shuffle=False, collate_fn=coco_dataset.collate_fn)
    for i, (ims, tgts) in enumerate(coco_dataloader):
        try:
            print(i, ims.shape, tgts.shape)
        except:
            import pdb; pdb.set_trace()