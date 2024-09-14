from typing import Any
import torchvision.transforms as transforms
import torch

class ToTensor(object):
    def __init__(self, ):
        pass

    def __call__(self, data):
        img, boxes = data
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(img)
        try:
            bb_targets = torch.zeros((len(boxes), 6))
            bb_targets[:, 1:] = transforms.ToTensor()(boxes)
        except:
            import pdb; pdb.set_trace()
        return img, bb_targets
    
class Resize(object):
    def __init__(self) -> None:
        pass

    def __call__(self, data) -> Any:
        img, boxes = data