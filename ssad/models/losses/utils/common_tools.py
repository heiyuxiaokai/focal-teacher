#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import warnings

import torch
from mmcv.ops import nms_rotated
from mmrotate.core.visualization import imshow_det_rbboxes

INF = 1e8

def get_rboxes(bboxes, scores, centerness, points, 
               score_thr=0.1, iou_thr=0.1, single_class=False):
    # obtain det_bboxes after post-progressing
    scores_for_thr, labels_for_thr = scores.sigmoid().max(dim=1)
    scores_ = scores_for_thr * centerness.sigmoid()
    inds = torch.nonzero(scores_for_thr >= score_thr).squeeze(-1)
        
    socres_for_nms = scores_[inds]
    labels_for_nms = labels_for_thr[inds]
    bboxes_for_nms = bboxes[inds]
        
    if bboxes_for_nms.numel() == 0:
        return scores.new_zeros((0, 5)), scores.new_zeros((0)), \
               scores.new_zeros((0)), scores.new_zeros((0))

    if not single_class:
        _, keep = nms_rotated(bboxes_for_nms, socres_for_nms, iou_thr, labels_for_nms)
    else:
        _, keep = nms_rotated(bboxes_for_nms, socres_for_nms, iou_thr)
        
    keep = keep[:2000]
    det_bboxes = bboxes_for_nms[keep]
    det_labels = labels_for_nms[keep]
    det_scores = scores_for_thr[inds[keep]]
    nms_keep = inds[keep]

    return det_bboxes, det_scores, det_labels, nms_keep

def fcos_sample(gt_bboxes, 
                points, 
                regress_ranges, 
                num_points_per_lvl,
                strides=[8, 16, 32, 64, 128],
                center_sampling=True,
                center_sample_radius=1.5):
    num_points = points.size(0)
    num_gts = gt_bboxes.size(0)
    if num_gts == 0:
        min_area = gt_bboxes.new_zeros((num_points)).fill_(INF)
        min_area_inds = gt_bboxes.new_zeros((num_points))
        bbox_targets = gt_bboxes.new_zeros((num_points))
        return (min_area, min_area_inds), bbox_targets

    areas = gt_bboxes[:, 2] * gt_bboxes[:, 3]
    # TODO: figure out why these two are different
    # areas = areas[None].expand(num_points, num_gts)
    areas = areas[None].repeat(num_points, 1)
    regress_ranges = regress_ranges[:, None, :].expand(
        num_points, num_gts, 2)
    points = points[:, None, :].expand(num_points, num_gts, 2)
    gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)
    gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

    cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
    rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                            dim=-1).reshape(num_points, num_gts, 2, 2)
    offset = points - gt_ctr
    offset = torch.matmul(rot_matrix, offset[..., None])
    offset = offset.squeeze(-1)

    w, h = gt_wh[..., 0], gt_wh[..., 1]
    offset_x, offset_y = offset[..., 0], offset[..., 1]
    left = w / 2 + offset_x
    right = w / 2 - offset_x
    top = h / 2 + offset_y
    bottom = h / 2 - offset_y
    bbox_targets = torch.stack((left, top, right, bottom), -1)

    # condition1: inside a gt bbox
    inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
    if center_sampling:
        # condition1: inside a `center bbox`
        radius = center_sample_radius
        stride = offset.new_zeros(offset.shape)

        # project the points on current lvl back to the `original` sizes
        lvl_begin = 0
        for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
            lvl_end = lvl_begin + num_points_lvl
            stride[lvl_begin:lvl_end] = strides[lvl_idx] * radius
            lvl_begin = lvl_end

        inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
        inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

    # condition2: limit the regression range for each location
    max_regress_distance = bbox_targets.max(-1)[0]
    inside_regress_range = (
        (max_regress_distance >= regress_ranges[..., 0])
        & (max_regress_distance <= regress_ranges[..., 1]))

    # if there are still more than one objects for a location,
    # we choose the one with minimal area
    areas[inside_gt_bbox_mask == 0] = INF
    areas[inside_regress_range == 0] = INF
    min_area, min_area_inds = areas.min(dim=1)

    # labels = gt_labels[min_area_inds]
    # labels[min_area == INF] = self.num_classes  # set as BG
    bbox_targets = bbox_targets[range(num_points), min_area_inds]
    angle_targets = gt_angle[range(num_points), min_area_inds]

    return (min_area, min_area_inds), \
           torch.cat([bbox_targets, angle_targets], dim=-1)

def centerness_target(pos_bbox_targets):
    """Compute centerness targets.

    Args:
        pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
            (num_pos, 4)
    Returns:
        Tensor: Centerness target.
    """
    # only calculate pos centerness targets, otherwise there may be nan
    left_right = pos_bbox_targets[:, [0, 2]]
    top_bottom = pos_bbox_targets[:, [1, 3]]
    if len(left_right) == 0:
        centerness_targets = left_right[..., 0]
    else:
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness_targets)

def detection_log_iter(image, tdets, tlabels, sdets, slabels, 
                       iter_count, class_names, iter_step):
    root_dir = os.environ.get("WORK_DIR", ".")
    # print(root_dir)
    # print(root_dir)
    if iter_count % iter_step == 0:
        filename = str(iter_count) + '.png'
        # print(os.path.join(root_dir, 'pseudo_boxes_teacher', filename))
        imshow_det_rbboxes(
            image,
            tdets,
            tlabels,
            class_names=class_names,
            show=False,
            out_file=os.path.join(root_dir, 'pseudo_boxes_teacher', filename),
        )

        imshow_det_rbboxes(
            image,
            sdets,
            slabels,
            class_names=class_names,
            show=False,
            out_file=os.path.join(root_dir, 'pseudo_boxes_student', filename),
        )
