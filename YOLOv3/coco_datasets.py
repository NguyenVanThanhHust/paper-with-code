from os.path import join, isdir, isfile
from pycocotools.coco import COCO
import numpy as np
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
try:
    from transform import ToTensor
except:
    from data.transform import ToTensor
    
class CocoDetectionDataset(Dataset):
    def __init__(self, image_folder, annFile, transform=None) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.coco = COCO(annFile)
        self.img_ids = self.coco.getImgIds()
        self.transform = transform
        print(self.transform)

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
            img, bboxes = self.transform((img, bboxes))
        return img, bboxes
    

    
if __name__ == '__main__':
    # coco_dataset = CocoDetectionDataset(image_folder="/home/thanhnv154te/workspace/Datasets/COCO/train2017", annFile="/home/thanhnv154te/workspace/Datasets/COCO/annotations_trainval2017/annotations/instances_train2017.json")
    val_transform = transforms.Compose([
        ToTensor()
    ])
    
    coco_dataset = CocoDetectionDataset(image_folder="/home/thanhnv154te/workspace/Datasets/COCO/val2017", 
                                        annFile="/home/thanhnv154te/workspace/Datasets/COCO/annotations_trainval2017/annotations/instances_val2017.json",
                                        transform=val_transform)
    print(len(coco_dataset))
    coco_dataloader = DataLoader(coco_dataset, 2, False)
    for ims, tgts in coco_dataloader:
        print(ims.shape)
        print(tgts.shape)
        break