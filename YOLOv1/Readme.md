# YOLO V1
This is my implementation of YOLO v1

## How to run
Download dataset from [pascal-voc-2012-dataset](https://www.kaggle.com/datasets/gopalbhattrai/pascal-voc-2012-dataset)

To train PASCAL VOC 2012
```
python main.py 
```

To train COCO
```
python main.py --data_set COCO --S 7 --C 80 -bs 2
```

## Reference
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/YOLO