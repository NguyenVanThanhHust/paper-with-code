import numpy as np
import cv2

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
    
