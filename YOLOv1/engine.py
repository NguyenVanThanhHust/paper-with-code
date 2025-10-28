import torch
from loguru import logger

from utils import mean_average_precision, prediction_tensor_to_boxes, target_tensor_to_boxes

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch=0):
    model.train()
    model = model.to(device)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        logger.info("Epoch: {} Batch: {}/{} Loss: {}".format(epoch, batch_idx, len(dataloader), loss.item()))        

def evaluate_tmp(model, dataloader, criterion, device, epoch):
    model.eval()
    model = model.to(device)
    losses = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            logger.info("Epoch: {} Batch: {}/{} Loss: {}".format(epoch, batch_idx, len(dataloader), loss.item()))
            losses.append(loss.item())
    average_loss = sum(losses) / len(losses)        
    path = f"yolov1_{epoch}_{average_loss:0.4f}.pth"
    print(f"Saving: {path}")
    torch.save(model.state_dict(), path)


def evaluate(model, dataloader, device, epoch):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred_boxes = prediction_tensor_to_boxes(outputs, model.split_size, model.num_boxes, model.num_classes)
            tgt_boxes = target_tensor_to_boxes(targets, model.split_size, model.num_classes)
            mAP = mean_average_precision(pred_boxes, tgt_boxes, num_classes=model.num_classes)
            logger.info(f"Batch: {batch_idx} mAP: {mAP}")        


def infer():
    return