#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/9/18 21:01
# @Author : WeiHua
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import nms_rotated
from mmrotate.models import ROTATED_LOSSES, build_loss
from mmrotate.core import build_bbox_coder, rbbox_overlaps, norm_angle, obb2poly, poly2obb
from mmdet.core import reduce_mean
from mmdet.core.anchor.point_generator import MlvlPointGenerator
from .utils.common_tools import detection_log_iter

INF = 1e8

def postprogress(
        bboxes, scores, centerness, score_thr=0.3, 
        iou_thr=0.1, single_class=False
    ):
    # obtain det_bboxes after post-progressing
    scores_for_thr, labels_for_thr = scores.sigmoid().max(dim=1)
    scores_ = scores_for_thr * centerness.sigmoid()
    inds = torch.nonzero(scores_for_thr >= score_thr).squeeze(-1)
        
    socres_for_nms = scores_[inds]
    labels_for_nms = labels_for_thr[inds]
    bboxes_for_nms = bboxes[inds]
        
    if bboxes_for_nms.numel() == 0:
        return scores.new_zeros((0))

    if not single_class:
        _, keep = nms_rotated(bboxes_for_nms, socres_for_nms, iou_thr, labels_for_nms)
    else:
        _, keep = nms_rotated(bboxes_for_nms, socres_for_nms, iou_thr)
        
    keep = keep[:2000]
    nms_keep = inds[keep]

    return nms_keep

@ROTATED_LOSSES.register_module()
class RotatedFTLoss(nn.Module):
    def __init__(self, cls_channels=16, loss_type='origin', bbox_loss_type='l1'):
        super(RotatedFTLoss, self).__init__()
        self.cls_channels = cls_channels
        assert bbox_loss_type in ['l1', 'iou']
        self.bbox_loss_type = bbox_loss_type
        if self.bbox_loss_type == 'l1':
            self.bbox_loss = nn.SmoothL1Loss(reduction='none')
        else:
            self.bbox_loss = build_loss(dict(type='RotatedIoULoss'))
        
        # self.bbox_coder = build_bbox_coder(dict(type='DistanceAnglePointCoder', angle_version='le90'))
        # self.prior_generator = MlvlPointGenerator([8, 16, 32, 64, 128])
        # self.cls_loss = build_loss(dict(type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25,))
        # self.centerness_loss = build_loss(dict(type='CrossEntropyLoss', use_sigmoid=True))
        # self.loss_type = loss_type
        # self.strides = [8, 16, 32, 64, 128]

    def convert_shape(self, logits):
        cls_scores, bbox_preds, angle_preds, centernesses = logits
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(centernesses)

        batch_size = cls_scores[0].shape[0]
        cls_scores = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1, self.cls_channels) for x in cls_scores
        ], dim=1)
        bbox_preds = torch.cat([
            torch.cat([x, y], dim=1).permute(0, 2, 3, 1).reshape(batch_size, -1, 5) for x, y in
            zip(bbox_preds, angle_preds)
        ], dim=1)
        centernesses = torch.cat([
            x.permute(0, 2, 3, 1).reshape(batch_size, -1) for x in centernesses
        ], dim=1)
        return cls_scores, bbox_preds, centernesses

    def forward(self, teacher_logits, student_logits, student_labels, iter_count, class_names, ratio=0.01, img_metas=None,
                **kwargs):
        """

        Args:
            teacher_logits (Tuple): logits of teacher, with keys:
                "cls_score": list[Tensor], Box scores for each scale level, each is a 4D-tensor,
                the channel number is num_points * num_classes.
                "bbox_pred": list[Tensor], Box energies / deltas for each scale level, each is a 4D-tensor,
                the channel number is num_points * 4.
                "angle_pred": list[Tensor], Box angle for each scale level, each is a 4D-tensor,
                the channel number is num_points * 1.
                "centerness": list[Tensor], centerness for each scale level, each is a 4D-tensor,
                the channel number is num_points * 1.
            student_logits (Tuple): logits of student, same as teacher.
            ratio (float): sampling ratio for loss calculation
            img_metas (Optional | Dict): img metas

        Returns:

        """
        # print(img_metas)
        device = teacher_logits[0][0].device
        t_cls_scores, t_bbox_preds, t_centernesses = self.convert_shape(teacher_logits)
        s_cls_scores, s_bbox_preds, s_centernesses = self.convert_shape(student_logits)
        
        # # points
        # all_level_points = self.prior_generator.grid_priors(
        #     [featmap.size()[-2:] for featmap in teacher_logits[0]],
        #     dtype=s_bbox_preds.dtype,
        #     device=s_bbox_preds.device)
        # num_points_per_lvl = [points.shape[0] for points in all_level_points]
        # flatten_points = torch.cat(all_level_points)
        
        # # regress_ranges
        # regress_ranges = ((-1, 64), (64, 128), (128, 256), (256, 512), (512, INF))
        # all_level_regress_ranges = [
        #     all_level_points[i].new_tensor(regress_ranges[i])[None].expand_as(
        #         all_level_points[i]) for i in range(len(all_level_points))
        # ]
        # flatten_regress_ranges = torch.cat(all_level_regress_ranges)

        # # strides
        # all_level_strides = [
        #     all_level_points[i][:, 0].new_tensor(self.strides[i])[None].expand_as(
        #         all_level_points[i][:, 0]) for i in range(len(all_level_points))
        # ]
        # flatten_strides = torch.cat(all_level_strides)
        unsup_loss_cls, unsup_loss_reg, unsup_loss_quality = [], [], []
        for img_idx in range(t_cls_scores.shape[0]):
            t_cls_score, t_bbox_pred, t_centerness = \
                t_cls_scores[img_idx], t_bbox_preds[img_idx], t_centernesses[img_idx]
            s_cls_score, s_bbox_pred, s_centerness = \
                s_cls_scores[img_idx], s_bbox_preds[img_idx], s_centernesses[img_idx]

            # m_strides = torch.cat([
            #     flatten_strides.unsqueeze(-1).repeat(1, 4), 
            #     torch.ones_like(flatten_strides).unsqueeze(-1)], dim=1)

            # t_rbbox_pred = self.bbox_coder.decode(
            #     flatten_points, t_bbox_pred * m_strides
            # ) #(-1, 5)
            
            # s_rbbox_pred = self.bbox_coder.decode(
            #     flatten_points, s_bbox_pred * m_strides
            # ) #(-1, 5)
            
            # nms_keep = postprogress(t_rbbox_pred, t_cls_score, t_centerness)
            max_vals, max_labels = torch.max(t_cls_score.sigmoid(), 1)
            
            with torch.no_grad():
                weight = max_vals
                pos_factor = (max_vals * t_centerness.sigmoid())

            # cls
            t_cls_score_sigmoid = t_cls_score.sigmoid()
            pt = (s_cls_score.sigmoid() - t_cls_score_sigmoid).abs()
            cls_loss = pt ** 2.0 * F.binary_cross_entropy(
                s_cls_score.sigmoid(), t_cls_score_sigmoid, reduction='none'
            )
            unsup_loss_cls_per_img = cls_loss.sum() / (pos_factor ** 1.0).sum()

            lamada = 2.
            # reg
            reg_prob = torch.exp(
                -2. * self.bbox_loss(s_bbox_pred, t_bbox_pred).sum(dim=-1)
            )
            ptr = weight * (1 - reg_prob)
            # unsup_loss_reg_per_img = (ptr ** lamada * F.binary_cross_entropy(
            #     ptr, torch.zeros_like(reg_prob), reduction='none'
            # )).sum() / (pos_factor ** 1.0).sum()
            unsup_loss_reg_per_img = (-1. * ptr ** lamada * torch.log(
                1 - ptr + 1e-12
            )).sum() / (pos_factor ** 1.0).sum()

            # cen
            ptc =  weight * (s_centerness.sigmoid() - t_centerness.sigmoid()).abs()
            # unsup_loss_qua_per_img = (ptc ** lamada * F.binary_cross_entropy(
            #     ptc, torch.zeros_like(ptc), reduction='none'
            # )).sum() / (pos_factor ** 1.0).sum()
            unsup_loss_qua_per_img = (-1. * ptc ** lamada * torch.log(
                1 - ptc + 1e-12
            )).sum() / (pos_factor ** 1.0).sum()
            # print((-1. * ptc ** lamada * torch.log(1 - ptc + 1e-12)).sum(), (ptc ** 1.0).sum())
            
            unsup_loss_cls.append(unsup_loss_cls_per_img)
            unsup_loss_reg.append(unsup_loss_reg_per_img)
            unsup_loss_quality.append(unsup_loss_qua_per_img)
        
        unsup_loss_cls = torch.stack(unsup_loss_cls).mean()
        unsup_loss_reg = torch.stack(unsup_loss_reg).mean()
        unsup_loss_quality = torch.stack(unsup_loss_quality).mean()

        unsup_losses = dict(
            loss_cls=unsup_loss_cls,
            loss_bbox=unsup_loss_reg,
            loss_centerness=unsup_loss_quality
        )

        return unsup_losses
