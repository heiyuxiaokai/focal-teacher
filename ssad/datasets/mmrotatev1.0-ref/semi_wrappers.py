# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Callable, Dict, List, Optional, Union

import numpy as np
from mmcv.transforms import BaseTransform, Compose

from mmrotate.registry import TRANSFORMS

@TRANSFORMS.register_module()
class STMultiBranch(BaseTransform):
    r"""Multiple branch pipeline wrapper.

    Generate multiple data-augmented versions of the same image.
    `MultiBranch` needs to specify the branch names of all
    pipelines of the dataset, perform corresponding data augmentation
    for the current branch, and return None for other branches,
    which ensures the consistency of return format across
    different samples.

    Args:
        branch_field (list): List of branch names.
        branch_pipelines (dict): Dict of different pipeline configs
            to be composed.
    """

    def __init__(self, branch_field: List[str],
                 **branch_pipelines: dict) -> None:
        self.branch_field = branch_field
        self.branch_pipelines = {
            branch: Compose(pipeline)
            for branch, pipeline in branch_pipelines.items()
        }

    def transform(self, results: dict) -> dict:
        """Transform function to apply transforms sequentially.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict:

            - 'inputs' (Dict[str, obj:`torch.Tensor`]): The forward data of
                models from different branches.
            - 'data_sample' (Dict[str,obj:`DetDataSample`]): The annotation
                info of the sample from different branches.
        """
        multi_results = {}
        for branch in self.branch_field:
            multi_results[branch] = {'inputs': None, 'data_samples': None}
        
        if "sup" in self.branch_pipelines.keys():
            pipeline = self.branch_pipelines["sup"]
            branch_results = pipeline(copy.deepcopy(results))
            multi_results["sup"] = branch_results
        
        if ("unsup_teacher" in self.branch_pipelines.keys()) and \
               ("unsup_student" in self.branch_pipelines.keys()) and \
               ("common" in self.branch_pipelines.keys()):

            weak_pipeline = self.branch_pipelines["unsup_teacher"]
            strong_pipeline = self.branch_pipelines["unsup_student"]
            common_pipeline = self.branch_pipelines["common"]

            teacher_results = weak_pipeline(copy.deepcopy(results))
            student_results = strong_pipeline(copy.deepcopy(teacher_results))
            
            multi_results["unsup_teacher"] = common_pipeline(copy.deepcopy(teacher_results))
            multi_results["unsup_student"] = common_pipeline(copy.deepcopy(student_results))

            
            # import cv2
            # import random
            # path = '/media/fw/wangkai/Codesource/rs-ssod/work_dirs/tem/'
            # rseed = str(random.random())
            # img_name1 = path + rseed + 'teacher_' + '.png'
            # img_name2 = path + rseed + 'student_' + '.png'
            # print(multi_results["unsup_teacher"]['inputs'].shape)
            # cv2.imwrite(img_name1, multi_results["unsup_teacher"]['inputs'].permute(2, 1, 0).cpu().numpy())
            # cv2.imwrite(img_name2, multi_results["unsup_student"]['inputs'].permute(2, 1, 0).cpu().numpy())

            
        format_results = {}
        for branch, results in multi_results.items():
            for key in results.keys():
                if format_results.get(key, None) is None:
                    format_results[key] = {branch: results[key]}
                else:
                    format_results[key][branch] = results[key]
        return format_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(branch_pipelines={list(self.branch_pipelines.keys())})'
        return repr_str