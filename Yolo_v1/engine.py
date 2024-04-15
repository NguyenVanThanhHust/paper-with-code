import torch
from loguru import logger

from utils import mean_average_precision

def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch=0):
    model.train()
    model = model.to(device)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        logger.info("Epoch: {} Batch: {}/{} Loss: {}".format(epoch, batch_idx, len(dataloader), loss.item()))        
    return

def evaluate(model, dataloader, device, epoch):
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            pred_boxes = cell_to_boxes(outputs, model.)
            tgt_boxes = cell_to_boxes(targets)
            import pdb; pdb.set_trace()
    return

def infer():
    return