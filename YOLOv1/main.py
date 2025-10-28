import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from loguru import logger

from losses import build_loss
from model import build_model
from engine import train_one_epoch, evaluate, evaluate_tmp

def get_args():
    parser = argparse.ArgumentParser("YOLO v1")
    parser.add_argument('--data_folder', type=str, default="../../Datasets/PASCAL_VOC/")
    parser.add_argument('--data_set', type=str, choices=["VOC", "COCO"], default="VOC")
    
    parser.add_argument('--S', type=int, default=7, help="number of cell  in each dim")
    parser.add_argument('--B', type=int, default=2, help="number of prediction in each cell")
    parser.add_argument('--C', type=int, default=20, help='number of class in dataset')

    parser.add_argument('-bs', '--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='number of worker for data loader')
    parser.add_argument('--epochs', type=int, default=2, help='number of worker for data loader')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')

    parser.add_argument('--task', choices=['train', 'evaluate', 'infer'], default='train', help='task to be done')
    parser.add_argument('--pretrain_ckpt', type=str, help='pretrained checkpoint path')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.data_set == "VOC":
        from pascal_voc import build_dataloader
    else:
        from coco_datasets import build_dataloader    
    train_dataloader = build_dataloader(args, split="train")
    val_dataloader = build_dataloader(args, split="val")
    model = build_model(args)
    criterion = build_loss(args)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                        momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.task == 'train':
        logger.info("Start to train")
        for epoch in range(args.epochs):
            train_one_epoch(model, train_dataloader,criterion, optimizer, scheduler, device, epoch)
            # evaluate(model, val_dataloader, device, epoch)
            evaluate_tmp(model, val_dataloader, criterion, device, epoch)
    elif args.task == 'evaluate':
        logger.info("Start to evaluate")
        evaluate(model, val_dataloader, device, 0)
    else:
        logger.info("Start inference")
