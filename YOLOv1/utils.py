import os, sys
import torch
import numpy as np

from collections import Counter

def convert_prediction_tensor(predictions, split_size:int=7, num_boxes: int=2, num_classes:int=20):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, split_size, split_size, num_classes + num_boxes*5)
    bboxes1 = predictions[..., num_classes+1:num_classes+5]
    bboxes2 = predictions[..., num_classes+6:num_classes+10]
    scores = torch.cat(
        (predictions[..., num_classes].unsqueeze(0), predictions[..., num_classes+num_boxes].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(split_size).repeat(batch_size, split_size, 1).unsqueeze(-1)
    x = 1 / split_size * (best_boxes[..., :1] + cell_indices)
    y = 1 / split_size * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / split_size * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :num_classes].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., num_classes], predictions[..., num_classes+num_boxes]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def prediction_tensor_to_boxes(out, split_size:int=7, num_boxes: int=2, num_classes:int=20):
    converted_pred = convert_prediction_tensor(out, split_size, num_boxes, num_classes).reshape(out.shape[0], split_size * split_size, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(split_size * split_size):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def target_tensor_to_boxes(tgts, split_size:int=7, num_classes:int=20):
    batch_size = tgts.shape[0]
    all_bboxes = []
    for b in range(batch_size):
        tgt_tensor = tgts[b]
        cls_tensor = tgt_tensor[:, :, :num_classes]
        positions = torch.nonzero(cls_tensor == 1)
        positions = positions.detach().cpu().numpy().tolist()
        boxes = []
        for position in positions:
            row, col, cls_idx = position
            box_tensor = tgt_tensor[row, col, -4:]
            offset_cx, offset_cy, scaled_w, scaled_h  = box_tensor.detach().cpu().numpy().tolist()
            cx = (offset_cx + col) / split_size
            cy = (offset_cy + row) / split_size
            w, h = scaled_w / split_size, scaled_h / split_size
            box = [cls_idx, cx, cy, w, h]
            boxes.append(box)
        all_bboxes.append(boxes)
    return all_bboxes

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor, format="cxcywh") -> torch.Tensor:
    """

    """
    assert format in ["cxcywh", "xyxy", "xywh"]
    if format == "cxcywh":
        box1_x1 = box1[..., 0] - box1[..., 2] / 2
        box1_y1 = box1[..., 1] - box1[..., 3] / 2
        box1_x2 = box1[..., 0] + box1[..., 2] / 2
        box1_y2 = box1[..., 1] + box1[..., 3] / 2

        box2_x1 = box2[..., 0] - box2[..., 2] / 2
        box2_y1 = box2[..., 1] - box2[..., 3] / 2
        box2_x2 = box2[..., 0] + box2[..., 2] / 2
        box2_y2 = box2[..., 1] + box2[..., 3] / 2
        box1_area = box1[..., 2] * box1[..., 3]
        box2_area = box2[..., 2] * box2[..., 3]
    elif format == "xyxy":
        box1_x1 = box1[..., 0]
        box1_y1 = box1[..., 1]
        box1_x2 = box1[..., 2]
        box1_y2 = box1[..., 3]

        box2_x1 = box2[..., 0]
        box2_y1 = box2[..., 1]
        box2_x2 = box2[..., 2]
        box2_y2 = box2[..., 3]
        box1_area = (box1[..., 2]-box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2]-box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    else:
        box1_x1 = box1[..., 0]
        box1_y1 = box1[..., 1]
        box1_x2 = box1[..., 0] + box1[..., 2]
        box1_y2 = box1[..., 1] + box1[..., 3]

        box2_x1 = box2[..., 0]
        box2_y1 = box2[..., 1]
        box2_x2 = box2[..., 0] + box2[..., 2]
        box2_y2 = box2[..., 1] + box2[..., 3]
        box1_area = box1[..., 2] * box1[..., 3]
        box2_area = box2[..., 2] * box2[..., 3]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU) 
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
            < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    average_precisions = []
    for c in range(num_classes):
        # used for numerical stability later on
        epsilon = 1e-6
        gts = []
        preds = []
        for true_box in true_boxes:
            if int(true_box[1]) != c:
                continue
            gts.append(true_box)
        for pred_box in pred_boxes:
            if int(pred_box[1]) != c:
                continue
            preds.append(pred_box)
        preds.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(preds)))
        FP = torch.zeros((len(preds)))
        num_gts = len(gts)
        gt_ids = np.zeros(len(gts))
        for pred_idx, pred_box in enumerate(preds):
            image_id = int(pred_box[0])
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt_box in enumerate(gts):
                if int(gt_box[0]) != image_id or gt_ids[gt_idx] == 1:
                    continue
                iou = calculate_iou(torch.from_numpy(np.array(pred_box[-4:])), torch.from_numpy(np.array(gt_box[-4:])), format="xywh")
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            if best_iou > iou_threshold:
                TP[pred_idx] = 1
                gt_ids[best_gt_idx] = 1
            else:
                FP[pred_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (num_gts + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
    print(average_precisions)
    return sum(average_precisions) / len(average_precisions)

if __name__ == "__main__":
    ## each box has format
    ## gt: [image_id, class_idx, x1, y1, x2, y2]
    ## pred: [image_id, class_idx, confidence score, x1, y1, x2, y2]
    ## box is scaled to range (0, 1)
    gts = [
        [397133, 44, 0.34003125, 0.5633255269320843, 0.400953125, 0.6985714285714285],
        [397133, 67, 0.0015625, 0.5626229508196722, 0.543171875, 1.0],
        [397133, 1, 0.60728125, 0.1637470725995316, 0.778234375, 0.8139110070257611],
        [397133, 49, 0.21182812499999998, 0.5841451990632318, 0.24670312499999997, 0.651569086651054],
        [397133, 51, 0.048875, 0.8056206088992974, 0.15531250000000002, 0.9012412177985948],
        [397133, 51, 0.093171875, 0.6729742388758783, 0.21203124999999998, 0.7696955503512881],
        [397133, 79, 0.002125, 0.38484777517564406, 0.30300000000000005, 0.6152224824355973],
        [397133, 1, 0.0, 0.615480093676815, 0.09712499999999999, 0.7015925058548009],
        [397133, 47, 0.18656250000000002, 0.6381967213114754, 0.22534375, 0.718407494145199],
        [397133, 47, 0.221046875, 0.6274238875878221, 0.27134375, 0.7114051522248245],
        [397133, 51, 0.243703125, 0.39566744730679154, 0.284375, 0.4357845433255269],
        [397133, 51, 0.24562499999999998, 0.26733021077283375, 0.27353125, 0.30437939110070256],
        [397133, 56, 0.154296875, 0.7137704918032787, 0.171140625, 0.726814988290398],
        [397133, 50, 0.259421875, 0.6003747072599532, 0.273203125, 0.643887587822014],
        [397133, 56, 0.135015625, 0.6884543325526933, 0.172453125, 0.7146370023419204],
        [397133, 56, 0.10959375, 0.6935831381733022, 0.12409375, 0.7043091334894613],
        [397133, 79, 0.0, 0.49391100702576113, 0.29900000000000004, 0.7257142857142858],
        [397133, 57, 0.151078125, 0.6957611241217798, 0.163328125, 0.7071428571428571],
        [397133, 81, 0.776953125, 0.4763466042154567, 0.96759375, 0.5433489461358314],
        [37777, 64, 0.2911647727272727, 0.5150869565217391, 0.31360795454545454, 0.5903478260869566],
        [37777, 62, 0.07528409090909091, 0.9358695652173913, 0.25, 0.9989130434782608],
        [37777, 62, 0.3309659090909091, 0.8242173913043478, 0.47301136363636365, 0.9350869565217391],
        [37777, 67, 0.22599431818181817, 0.7741304347826088, 0.8179261363636364, 0.9858695652173913],
        [37777, 82, 0.8574999999999999, 0.3258260869565217, 0.998465909090909, 0.9842608695652174],
        [37777, 52, 0.6268465909090909, 0.7779565217391304, 0.7343465909090909, 0.8993043478260869],
        [37777, 79, 0.39053977272727275, 0.539608695652174, 0.5615056818181818, 0.8483913043478261],
        [37777, 81, 0.7566193181818182, 0.5847391304347827, 0.8357954545454546, 0.5998260869565217],
        [37777, 55, 0.6129829545454546, 0.8039130434782609, 0.6620170454545454, 0.8733478260869565],
        [37777, 55, 0.6579545454545455, 0.8737391304347827, 0.7053693181818181, 0.9430000000000001],
        [37777, 55, 0.654971590909091, 0.775, 0.6882954545454546, 0.8227391304347825],
        [37777, 55, 0.5829261363636363, 0.8130869565217391, 0.6255113636363636, 0.9052608695652173],
        [37777, 62, 0.6873579545454546, 0.7844347826086956, 0.8332954545454545, 0.9818260869565217],
        [37777, 55, 0.6190056818181818, 0.872, 0.6591761363636363, 0.9306521739130436],
        [252219, 1, 0.5098125, 0.40785046728971963, 0.6211249999999999, 0.868714953271028],
        [252219, 1, 0.015296874999999998, 0.39032710280373834, 0.20582812499999997, 0.9194158878504672],
        [252219, 1, 0.7975625, 0.40016355140186916, 0.9907812500000001, 0.9042757009345794],
        [252219, 28, 0.876140625, 0.2108644859813084, 1.0, 0.36815420560747664],
        [252219, 31, 0.071890625, 0.4932476635514019, 0.1243125, 0.614392523364486],
        [252219, 47, 0.539265625, 0.5289953271028037, 0.556546875, 0.5807242990654206],
        [252219, 10, 0.52665625, 0.1030607476635514, 0.62253125, 0.2366355140186916],
    ]

    old_predictions = [
        [397133, 42,0.7579164869806583, 0.34003125, 0.5633255269320843, 0.400953125, 0.6985714285714285], # change class
        [397133, 67,0.03456578560102341, 0.0015625, 0.5626229508196722, 0.25, 0.75], # change x2 y2
        [397133, 1,0.019382678875603188, 0.60728125, 0.1637470725995316, 0.778234375, 0.8139110070257611],
        [397133, 49,0.6452423295841793, 0.21182812499999998, 0.5841451990632318, 0.24670312499999997, 0.651569086651054],
        [397133, 51,0.8696421401307523, 0.048875, 0.8056206088992974, 0.15531250000000002, 0.9012412177985948],
        [397133, 51,0.09393535638488892, 0.093171875, 0.6729742388758783, 0.21203124999999998, 0.7696955503512881],
        [397133, 79,0.6363522651605056, 0.002125, 0.38484777517564406, 0.30300000000000005, 0.6152224824355973],
        [397133, 1,0.7216532902425651, 0.0, 0.615480093676815, 0.09712499999999999, 0.7015925058548009],
        [397133, 47,0.2619976549467967, 0.18656250000000002, 0.6381967213114754, 0.22534375, 0.718407494145199],
        [397133, 56,0.7210052123238411, 0.154296875, 0.7137704918032787, 0.171140625, 0.726814988290398],
        [397133, 50,0.9662878331751061, 0.259421875, 0.6003747072599532, 0.273203125, 0.643887587822014],
        [397133, 56,0.10726784805836875, 0.135015625, 0.6884543325526933, 0.172453125, 0.7146370023419204],
        [397133, 56,0.6999221891697122, 0.10959375, 0.6935831381733022, 0.12409375, 0.7043091334894613],
        [397133, 79,0.8863459738437308, 0.0, 0.49391100702576113, 0.29900000000000004, 0.7257142857142858],
        [397133, 57,0.8034695619354654, 0.151078125, 0.6957611241217798, 0.163328125, 0.7071428571428571],
        [397133, 81,0.7122902258546552, 0.776953125, 0.4763466042154567, 0.96759375, 0.5433489461358314],
        [37777, 62,0.9158115632665459, 0.2911647727272727, 0.5150869565217391, 0.31360795454545454, 0.5903478260869566], # change cls
        [37777, 63,0.4216176419546831, 0.07528409090909091, 0.9358695652173913, 0.25, 0.9989130434782608], 
        [37777, 62,0.992986022011788, 0.3309659090909091, 0.8242173913043478, 0.47301136363636365, 0.9350869565217391],
        [37777, 52,0.5089500438111537, 0.6268465909090909, 0.7779565217391304, 0.7343465909090909, 0.8993043478260869],
        [37777, 79,0.7458354052499834, 0.39053977272727275, 0.539608695652174, 0.5615056818181818, 0.8483913043478261],
        [37777, 81,0.7779157292349476, 0.7566193181818182, 0.5847391304347827, 0.8357954545454546, 0.5998260869565217],
        [37777, 55,0.3693842574993753, 0.6129829545454546, 0.8039130434782609, 0.6620170454545454, 0.8733478260869565],
        [37777, 55,0.3150886191111919, 0.6579545454545455, 0.8737391304347827, 0.7053693181818181, 0.9430000000000001],
        [37777, 55,0.7675158044259299, 0.654971590909091, 0.775, 0.6882954545454546, 0.8227391304347825],
        [37777, 55,0.629169181340596, 0.2, 0.5, 0.6255113636363636, 0.9052608695652173],
        [37777, 62,0.5565947042238175, 0.3, 0.2, 0.8332954545454545, 0.9818260869565217],
        [37777, 55,0.7039238986036378, 0.1, 0.872, 0.6591761363636363, 0.9306521739130436],
        [252219, 1,0.4102643241002186, 0.5098125, 0.7, 0.6211249999999999, 0.868714953271028],
        [252219, 1,0.21444515294295552, 0.015296874999999998, 0.39032710280373834, 0.20582812499999997, 0.9194158878504672],
        [252219, 1,0.8700783463377121, 0.7975625, 0.40016355140186916, 0.9907812500000001, 0.9042757009345794],
        [252219, 28,0.5293906980385198, 0.876140625, 0.2108644859813084, 1.0, 0.36815420560747664],
        [252219, 31,0.3906665753009072, 0.071890625, 0.4932476635514019, 0.1243125, 0.614392523364486],
        [252219, 47,0.8063980125837075, 0.539265625, 0.5289953271028037, 0.556546875, 0.5807242990654206],
        [252219, 10,0.30515721867227563, 0.52665625, 0.1030607476635514, 0.62253125, 0.2366355140186916],
    ]

    predictions = [
        [397133, 44, 1.0, 0.34003125, 0.5633255269320843, 0.400953125, 0.6985714285714285],
        [397133, 67, 1.0, 0.0015625, 0.5626229508196722, 0.543171875, 1.0],
        [397133, 1, 1.0, 0.60728125, 0.1637470725995316, 0.778234375, 0.8139110070257611],
        [397133, 49, 1.0, 0.21182812499999998, 0.5841451990632318, 0.24670312499999997, 0.651569086651054],
        [397133, 51, 1.0, 0.048875, 0.8056206088992974, 0.15531250000000002, 0.9012412177985948],
        [397133, 51, 1.0, 0.093171875, 0.6729742388758783, 0.21203124999999998, 0.7696955503512881],
        [397133, 79, 1.0, 0.002125, 0.38484777517564406, 0.30300000000000005, 0.6152224824355973],
        [397133, 1, 1.0, 0.0, 0.615480093676815, 0.09712499999999999, 0.7015925058548009],
        [397133, 47, 1.0, 0.18656250000000002, 0.6381967213114754, 0.22534375, 0.718407494145199],
        [397133, 47, 1.0, 0.221046875, 0.6274238875878221, 0.27134375, 0.7114051522248245],
        [397133, 51, 1.0, 0.243703125, 0.39566744730679154, 0.284375, 0.4357845433255269],
        [397133, 51, 1.0, 0.24562499999999998, 0.26733021077283375, 0.27353125, 0.30437939110070256],
        [397133, 56, 1.0, 0.154296875, 0.7137704918032787, 0.171140625, 0.726814988290398],
        [397133, 50, 1.0, 0.259421875, 0.6003747072599532, 0.273203125, 0.643887587822014],
        [397133, 56, 1.0, 0.135015625, 0.6884543325526933, 0.172453125, 0.7146370023419204],
        [397133, 56, 1.0, 0.10959375, 0.6935831381733022, 0.12409375, 0.7043091334894613],
        [397133, 79, 1.0, 0.0, 0.49391100702576113, 0.29900000000000004, 0.7257142857142858],
        [397133, 57, 1.0, 0.151078125, 0.6957611241217798, 0.163328125, 0.7071428571428571],
        [397133, 81, 1.0, 0.776953125, 0.4763466042154567, 0.96759375, 0.5433489461358314],
        [37777, 64, 1.0, 0.2911647727272727, 0.5150869565217391, 0.31360795454545454, 0.5903478260869566],
        [37777, 62, 1.0, 0.07528409090909091, 0.9358695652173913, 0.25, 0.9989130434782608],
        [37777, 62, 1.0, 0.3309659090909091, 0.8242173913043478, 0.47301136363636365, 0.9350869565217391],
        [37777, 67, 1.0, 0.22599431818181817, 0.7741304347826088, 0.8179261363636364, 0.9858695652173913],
        [37777, 82, 1.0, 0.8574999999999999, 0.3258260869565217, 0.998465909090909, 0.9842608695652174],
        [37777, 52, 1.0, 0.6268465909090909, 0.7779565217391304, 0.7343465909090909, 0.8993043478260869],
        [37777, 79, 1.0, 0.39053977272727275, 0.539608695652174, 0.5615056818181818, 0.8483913043478261],
        [37777, 81, 1.0, 0.7566193181818182, 0.5847391304347827, 0.8357954545454546, 0.5998260869565217],
        [37777, 55, 1.0, 0.6129829545454546, 0.8039130434782609, 0.6620170454545454, 0.8733478260869565],
        [37777, 55, 1.0, 0.6579545454545455, 0.8737391304347827, 0.7053693181818181, 0.9430000000000001],
        [37777, 55, 1.0, 0.654971590909091, 0.775, 0.6882954545454546, 0.8227391304347825],
        [37777, 55, 1.0, 0.5829261363636363, 0.8130869565217391, 0.6255113636363636, 0.9052608695652173],
        [37777, 62, 1.0, 0.6873579545454546, 0.7844347826086956, 0.8332954545454545, 0.9818260869565217],
        [37777, 55, 1.0, 0.6190056818181818, 0.872, 0.6591761363636363, 0.9306521739130436],
        [252219, 1, 1.0, 0.5098125, 0.40785046728971963, 0.6211249999999999, 0.868714953271028],
        [252219, 1, 1.0, 0.015296874999999998, 0.39032710280373834, 0.20582812499999997, 0.9194158878504672],
        [252219, 1, 1.0, 0.7975625, 0.40016355140186916, 0.9907812500000001, 0.9042757009345794],
        [252219, 28, 1.0, 0.876140625, 0.2108644859813084, 1.0, 0.36815420560747664],
        [252219, 31, 1.0, 0.071890625, 0.4932476635514019, 0.1243125, 0.614392523364486],
        [252219, 47, 1.0, 0.539265625, 0.5289953271028037, 0.556546875, 0.5807242990654206],
        [252219, 10, 1.0, 0.52665625, 0.1030607476635514, 0.62253125, 0.2366355140186916],
    ]

    print(len(gts), len(predictions))
    mAP = mean_average_precision(pred_boxes=predictions, true_boxes=gts, num_classes=80)
    print(mAP)