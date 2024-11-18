import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch


def data_visualizer(img, idx_to_class, path, bbox=None, pred=None):
    """
    Data visualizer on the original image. Support both GT box input and proposal input.

    Input:
    - img: PIL Image input
    - idx_to_class: Mapping from the index (0-19) to the class name
    - bbox: GT bbox (in red, optional), a tensor of shape Nx5, where N is
            the number of GT boxes, 5 indicates (x_tl, y_tl, x_br, y_br, class)
    - pred: Predicted bbox (in green, optional), a tensor of shape N'x6, where
            N' is the number of predicted boxes, 6 indicates
            (x_tl, y_tl, x_br, y_br, class, object confidence score)
    """

    img_copy = np.array(img).astype('uint8')

    if bbox is not None:
        for bbox_idx in range(bbox.shape[0]):
            one_bbox = bbox[bbox_idx][:4].numpy().astype('int')
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                        one_bbox[3]), (255, 0, 0), 2)
            if bbox.shape[1] > 4: # if class info provided
                obj_cls = idx_to_class[bbox[bbox_idx][4].item()]
                cv2.putText(img_copy, '%s' % (obj_cls),
                            (one_bbox[0], one_bbox[1]+15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    if pred is not None:
        for bbox_idx in range(pred.shape[0]):
            one_bbox = pred[bbox_idx][:4].numpy().astype('int')
            cv2.rectangle(img_copy, (one_bbox[0], one_bbox[1]), (one_bbox[2],
                        one_bbox[3]), (0, 255, 0), 2)
            
            if pred.shape[1] > 4: # if class and conf score info provided
                obj_cls = idx_to_class[pred[bbox_idx][4].item()]
                conf_score = pred[bbox_idx][5].item()
                cv2.putText(img_copy, '%s, %.2f' % (obj_cls, conf_score),
                            (one_bbox[0], one_bbox[1]+15),
                            cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), thickness=1)

    plt.imshow(img_copy)
    plt.axis('off')
    plt.title(path)
    plt.savefig(path)
    plt.close()


def coord_trans(bbox, bbox_batch_idx, w_pixel, h_pixel, w_amap=7, h_amap=7, mode='a2p'):
    """
    Coordinate transformation function. It converts the box coordinate from
    the image coordinate system to the activation map coordinate system and vice versa.
    In our case, the input image will have a few hundred of pixels in
    width/height while the activation map is of size 7x7.

    Input:
    - bbox: Could be either bbox, anchor, or proposal, of shape Mx4
    - bbox_batch_idx: Index of the image that each bbox belongs to, of shape M
    - w_pixel: Number of pixels in the width side of the original image, of shape B
    - h_pixel: Number of pixels in the height side of the original image, of shape B
    - w_amap: Number of pixels in the width side of the activation map, scalar
    - h_amap: Number of pixels in the height side of the activation map, scalar
    - mode: Whether transfer from the original image to activation map ('p2a') or
            the opposite ('a2p')

    Output:
    - resized_bbox: Resized box coordinates, of the same shape as the input bbox
    """

    assert mode in ('p2a', 'a2p'), 'invalid coordinate transformation mode!'
    assert bbox.shape[-1] >= 4, 'the transformation is applied to the first 4 values of dim -1'

    if bbox.shape[0] == 0: # corner cases
        return bbox

    resized_bbox = bbox.clone()

    if mode == 'p2a':
        # pixel to activation
        width_ratio = w_pixel[bbox_batch_idx] * 1. / w_amap
        height_ratio = h_pixel[bbox_batch_idx] * 1. / h_amap
        resized_bbox[:, [0, 2]] /= width_ratio.view(-1, 1)
        resized_bbox[:, [1, 3]] /= height_ratio.view(-1, 1)
    else:
        # activation to pixel
        width_ratio = w_pixel[bbox_batch_idx] * 1. / w_amap
        height_ratio = h_pixel[bbox_batch_idx] * 1. / h_amap
        resized_bbox[:, [0, 2]] *= width_ratio.view(-1, 1)
        resized_bbox[:, [1, 3]] *= height_ratio.view(-1, 1)

    return resized_bbox


def generate_anchor(anc_per_grid, grid):
    """
    Anchor generator.

    Inputs:
    - anc_per_grid: Tensor of shape (A, 2) giving the shapes of anchor boxes to 
        consider at each point in the grid. anc_per_grid[a] = (w, h) gives the width
        and height of the a'th anchor shape.
    - grid: Tensor of shape (B, H', W', 2) giving the (x, y) coordinates of the
        center of each feature from the backbone feature map. This is the tensor
        returned from GenerateGrid.

    Outputs:
    - anchors: Tensor of shape (B, A, H', W', 4) giving the positions of all
        anchor boxes for the entire image. anchors[b, a, h, w] is an anchor box
        centered at grid[b, h, w], whose shape is given by anc[a]; we parameterize
        boxes as anchors[b, a, h, w] = (x_tl, y_tl, x_br, y_br), where (x_tl, y_tl)
        and (x_br, y_br) give the xy coordinates of the top-left and bottom-right
        corners of the box.
    """
    A, _ = anc_per_grid.shape
    B, H, W, _ = grid.shape
    anc_per_grid = anc_per_grid.to(grid)

    anc_per_grid = anc_per_grid.view(1, A, 1, 1, -1).repeat(B, 1, H, W, 1)
    grid = grid.view(B, 1, H, W, -1).repeat(1, A, 1, 1, 1)

    x1y1 = grid - anc_per_grid / 2
    x2y2 = grid + anc_per_grid / 2
    anchors = torch.cat([x1y1, x2y2], dim=-1)

    return anchors


def compute_iou(anchors, bboxes):
    """
    Compute the intersection-over-union between anchors and gts.
    
    Inputs:
    - anchors: Anchor boxes, of shape (M, 4), where M is the number of proposals
    - bboxes: GT boxes of shape (N, 4), where N is the number of GT boxes,
                4 indicates (x_{lr}^{gt}, y_{lr}^{gt}, x_{rb}^{gt}, y_{rb}^{gt})
    
    Outputs:
    - iou: IoU matrix of shape (M, N)
    """
    iou = None
    ##############################################################################
    # TODO: Given anchors and gt bboxes,                                         #
    # compute the iou between each anchor and gt bbox.                           #
    ##############################################################################
    x1 = torch.max(anchors[:, None, 0], bboxes[None, :, 0])
    y1 = torch.max(anchors[:, None, 1], bboxes[None, :, 1])
    x2 = torch.min(anchors[:, None, 2], bboxes[None, :, 2])
    y2 = torch.min(anchors[:, None, 3], bboxes[None, :, 3])
    inter = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area_anchors = (anchors[:, 2] - anchors[:, 0]) * (anchors[:, 3] - anchors[:, 1])
    area_bboxes = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    union = area_anchors[:, None] + area_bboxes[None, :] - inter
    iou = torch.zeros_like(union)
    non_zero_union = union > 0
    iou[non_zero_union] = inter[non_zero_union] / union[non_zero_union]
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return iou


def compute_offsets(anchors, bboxes):
    """
    Compute the offsets between anchors and gts.
    
    Inputs:
    - anchors: Anchor boxes, of shape (M, 4)
    - bboxes: GT boxes of shape (M, 4),
                4 indicates (x_{lr}^{gt}, y_{lr}^{gt}, x_{rb}^{gt}, y_{rb}^{gt})
    
    Outputs:
    - offsets: offsets of shape (M, 4)
    """
    wh_offsets = torch.log((bboxes[:, 2:4] - bboxes[:, :2]) \
        / (anchors[:, 2:4] - anchors[:, :2]))

    xy_offsets = (bboxes[:, :2] + bboxes[:, 2:4] - \
        anchors[:, :2] - anchors[:, 2:4]) / 2.

    xy_offsets /= (anchors[:, 2:4] - anchors[:, :2])

    offsets = torch.cat((xy_offsets, wh_offsets), dim=-1)

    return offsets


def generate_proposal(anchors, offsets):
    """
    Proposal generator.

    Inputs:
    - anchors: Anchor boxes, of shape (M, 4). Anchors are represented
    by the coordinates of their top-left and bottom-right corners.
    - offsets: Transformations of shape (M, 4) that will be used to
    convert anchor boxes into region proposals. The transformation
    offsets[m] = (tx, ty, tw, th) will be applied to the anchor
    anchors[m].

    Outputs:
    - proposals: Region proposals of shape (M, 4), represented by the
    coordinates of their top-left and bottom-right corners. Applying the
    transform offsets[m] to the anchor[m] should give the
    proposal proposals[m].

    """
    proposals = None
    ##############################################################################
    # TODO: Given anchor coordinates and the proposed offset for each anchor,    #
    # compute the proposal coordinates using the transformation formulas above.  #
    ##############################################################################
    # Replace "pass" statement with your code
    xy_offsets = offsets[:, :2]
    wh_offsets = offsets[:, 2:4]
    proposals_minus = torch.exp(wh_offsets) * (anchors[:, 2:4] - anchors[:, :2])
    proposals_plus = xy_offsets * (anchors[:, 2:4] - anchors[:, :2]) * 2 + (anchors[:, :2] + anchors[:, 2:4])
    proposals = torch.cat(((proposals_plus - proposals_minus) / 2, (proposals_plus + proposals_minus) / 2), dim=-1)
    ##############################################################################
    #                               END OF YOUR CODE                             #
    ##############################################################################

    return proposals


@torch.no_grad()
def assign_label(proposals, bboxes, background_id, pos_thresh=0.5, neg_thresh=0.5, pos_fraction=0.25):
    """
    Determine the activated (positive) and negative proposals for model training.

    For Fast R-CNN - Positive proposals are defined Any of the two
    (i) the proposal/proposals with the highest IoU overlap with a GT box, or
    (ii) a proposal that has an IoU overlap higher than positive threshold with any GT box.
    Note: One proposal can match at most one GT box (the one with the largest IoU overlapping).

    We assign a negative label to a proposal if its IoU ratio is lower than
    a threshold value for all GT boxes. Proposals that are neither positive nor negative
    do not contribute to the training objective.

    Main steps include:
    i) Decide activated and negative proposals based on the IoU matrix.
    ii) Compute GT confidence score/offsets/object class on the positive proposals.
    iii) Compute GT confidence score on the negative proposals.
    
    Inputs:
    - proposal: Proposal boxes, of shape (M, 4), where M is the number of proposals
    - bboxes: GT boxes of shape Nx5, where N is the number of GT boxes,
                5 indicates (x_{lr}^{gt}, y_{lr}^{gt}, x_{rb}^{gt}, y_{rb}^{gt}) and class index
    - background_id: Class id of the background class
    - pos_thresh: Positive threshold value
    - neg_thresh: Negative threshold value
    - pos_fraction: a factor balancing pos/neg proposals
    
    Outputs:
    - activated_anc_mask: a binary mask indicating the activated proposals, of shape M
    - negative_anc_mask: a binary mask indicating the negative proposals, of shape M
    - GT_class: GT class category on all proposals, background class for non-activated proposals,
                of shape M
    - bboxes: GT bboxes on activated proposals, of shape M'x4, where M' is the number of 
              activated proposals
    """
    M = proposals.shape[0]
    N = bboxes.shape[0]
    iou_mat = compute_iou(proposals, bboxes[:, :4])
    
    # activated/positive proposals
    max_iou_per_anc, max_iou_per_anc_ind = iou_mat.max(dim=-1)
    max_iou_per_box = iou_mat.max(dim=0, keepdim=True)[0]
    activated_anc_mask = (iou_mat == max_iou_per_box) & (max_iou_per_box > 0)
    activated_anc_mask |= (iou_mat > pos_thresh) # using the pos_thresh condition as well
    activated_anc_mask = activated_anc_mask.max(dim=-1)[0] # (M, )
    activated_anc_ind = torch.nonzero(activated_anc_mask.view(-1)).squeeze(-1)

    # GT class
    box_cls = bboxes[:, 4].long().view(1, N).expand(M, N)
    # if a proposal matches multiple GT boxes, choose the box with the largest iou
    GT_class = torch.gather(box_cls, -1, max_iou_per_anc_ind.unsqueeze(-1)).squeeze(-1) # M
    GT_class[~activated_anc_mask] = background_id

    # GT bboxes
    bboxes_expand = bboxes[:, :4].view(1, N, 4).expand((M, N, 4))
    bboxes = torch.gather(bboxes_expand, -2, max_iou_per_anc_ind.unsqueeze(-1) \
                          .unsqueeze(-1).expand(M, 1, 4)).view(M, 4)
    bboxes = bboxes[activated_anc_ind]

    # negative anchors
    negative_anc_mask = (max_iou_per_anc < neg_thresh)
    negative_anc_ind = torch.nonzero(negative_anc_mask.view(-1)).squeeze(-1)
    # balance pos/neg anchors, random choose
    num_neg = int(activated_anc_ind.shape[0] * (1 - pos_fraction) / pos_fraction)
    negative_anc_ind = negative_anc_ind[torch.randint(0, negative_anc_ind.shape[0], (num_neg,))]
    negative_anc_mask = torch.zeros_like(negative_anc_mask)
    negative_anc_mask[negative_anc_ind] = 1

    return activated_anc_mask, negative_anc_mask, GT_class, bboxes