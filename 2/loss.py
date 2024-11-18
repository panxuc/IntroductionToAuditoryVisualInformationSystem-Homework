import torch
import torch.nn.functional as F


def ClsScoreRegression(cls_scores, GT_label, batch_size):
    """
    Multi-class cross-entropy loss

    Inputs:
    - cls_scores: Predicted class scores, of shape (M, C).
    - GT_label: GT class labels, of shape (M,).

    Outputs:
    - cls_score_loss: Torch scalar
    """
    cls_loss = F.cross_entropy(cls_scores, GT_label, \
                                        reduction='sum') * 1. / batch_size
    return cls_loss


def BboxRegression(offsets, GT_offsets, batch_size):
    """"
    Use SmoothL1 loss as in Faster R-CNN

    Inputs:
    - offsets: Predicted box offsets, of shape (M, 4)
    - GT_offsets: GT box offsets, of shape (M, 4)

    Outputs:
    - bbox_reg_loss: Torch scalar
    """
    bbox_reg_loss = F.smooth_l1_loss(offsets, GT_offsets, reduction='sum') * 1. / batch_size
    return bbox_reg_loss