import math
import torch

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

def consistency_loss(
    reg_preds,
    aug_preds,
    reg_dims,
    aug_dims,
    const_data,
    topk_thresh = 5,
    iou_thresh = 0.25,
):
    classification_losses = []
    regression_losses = []
    batch_size = len(reg_preds)
    for j in range(batch_size):    
        # Retrieve classification and regression scores
        orig_classification = reg_preds[j]['instances'].get("scores")
        orig_regression = reg_preds[j]['instances'].get("pred_boxes")

        aug_classification = aug_preds[j]['instances'].get("scores")
        aug_regression = aug_preds[j]['instances'].get("pred_boxes")

        
        top_k = min(aug_classification.shape[0], min(orig_classification.shape[0], topk_thresh))
        # Exit early if no predictions to match
        if top_k == 0:
            continue
        top_orig_scores = torch.topk(orig_classification, top_k)[1]
        orig_scores = orig_classification[top_orig_scores]

        top_aug_scores = torch.topk(aug_classification, top_k)[1]
        aug_scores = aug_classification[top_aug_scores]

        # Get boxes corresponding to confident predictions
        orig_score_boxes = orig_regression[top_orig_scores].tensor
        aug_score_boxes = aug_regression[top_aug_scores].tensor

        # Get the scores for the confident box predictions
        orig_boxes_classification = orig_classification[top_orig_scores]
        aug_boxes_classification = aug_classification[top_aug_scores]

        #Find IoU between every original and augmented box prediction currently being considered
        IoU = calc_iou(orig_score_boxes, aug_score_boxes) #num orig boxes x num final boxes

        # Get max IoU for the ideal augmented box with the original box
        IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num orig boxes x 1

        # Find all indices where there's a significant enough overlap between the original and augmented prediction
        orig_aug_valid_overlap = torch.ge(IoU_max, iou_thresh)
    
        # Get corresponding augmented box for original boxes
        assigned_aug_boxes = aug_score_boxes[IoU_argmax, : ]
        assigned_aug_scores = aug_scores[IoU_argmax]

        # Filter down boxes to only ones with high overlap
        final_orig_boxes = orig_score_boxes[orig_aug_valid_overlap, :]
        assigned_aug_boxes = assigned_aug_boxes[orig_aug_valid_overlap, :]

        assiged_orig_scores = orig_scores[orig_aug_valid_overlap]
        assigned_aug_scores = assigned_aug_scores[orig_aug_valid_overlap]
        if final_orig_boxes.shape[0] == 0:
            continue
         
        orig_heights = final_orig_boxes[:, 3] - final_orig_boxes[:, 1]
        orig_widths = final_orig_boxes[:, 2] - final_orig_boxes[:, 0]
        orig_ctr_xs = final_orig_boxes[:, 0] + 0.5 * orig_widths
        orig_ctr_ys = final_orig_boxes[:, 1] + 0.5 * orig_heights

        aug_heights = assigned_aug_boxes[:, 3] - assigned_aug_boxes[:, 1]
        aug_widths = assigned_aug_boxes[:, 2] - assigned_aug_boxes[:, 0]
        aug_ctr_xs = assigned_aug_boxes[:, 0] + 0.5 * aug_widths
        aug_ctr_ys = assigned_aug_boxes[:, 1] + 0.5 * aug_heights


        reg_cols = reg_dims[j][1]
        reg_rows = reg_dims[j][2]
        aug_cols = aug_dims[j][1]
        aug_rows = aug_dims[j][2]

        x_difference = ((orig_ctr_xs - aug_ctr_xs)/reg_rows).square()
        y_difference = ((orig_ctr_ys - aug_ctr_ys)/reg_cols).square()
        w_difference = ((orig_widths - aug_widths)/reg_rows).square()
        h_difference = ((orig_heights - aug_heights)/reg_cols).square()

        regression_loss = 0.25 * (x_difference + y_difference + w_difference + h_difference)
        regression_losses.append(regression_loss.mean())
    final_regression_loss = torch.zeros(1, device=torch.device(reg_preds[0]['instances'].get("scores").get_device()))
    if len(regression_losses) > 0:
        final_regression_loss = torch.stack(regression_losses).mean(dim=0, keepdim=True)
    del regression_losses
    return final_regression_loss

def diou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Distance Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsct = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

    # Eqn. (7)
    loss = 1 - iou + (distance / diag_len)
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def ciou_loss(
    boxes1: torch.Tensor,
    boxes2: torch.Tensor,
    reduction: str = "none",
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Complete Intersection over Union Loss (Zhaohui Zheng et. al)
    https://arxiv.org/abs/1911.08287
    Args:
        boxes1, boxes2 (Tensor): box locations in XYXY format, shape (N, 4) or (4,).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
        eps (float): small number to prevent division by zero
    """

    x1, y1, x2, y2 = boxes1.unbind(dim=-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(dim=-1)

    # TODO: use torch._assert_async() when pytorch 1.8 support is dropped
    assert (x2 >= x1).all(), "bad box: x1 larger than x2"
    assert (y2 >= y1).all(), "bad box: y1 larger than y2"

    # Intersection keypoints
    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    intsct = torch.zeros_like(x1)
    mask = (ykis2 > ykis1) & (xkis2 > xkis1)
    intsct[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask])
    union = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - intsct + eps
    iou = intsct / union

    # smallest enclosing box
    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)
    diag_len = ((xc2 - xc1) ** 2) + ((yc2 - yc1) ** 2) + eps

    # centers of boxes
    x_p = (x2 + x1) / 2
    y_p = (y2 + y1) / 2
    x_g = (x1g + x2g) / 2
    y_g = (y1g + y2g) / 2
    distance = ((x_p - x_g) ** 2) + ((y_p - y_g) ** 2)

    # width and height of boxes
    w_pred = x2 - x1
    h_pred = y2 - y1
    w_gt = x2g - x1g
    h_gt = y2g - y1g
    v = (4 / (math.pi ** 2)) * torch.pow((torch.atan(w_gt / h_gt) - torch.atan(w_pred / h_pred)), 2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    # Eqn. (10)
    loss = 1 - iou + (distance / diag_len) + alpha * v
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss
