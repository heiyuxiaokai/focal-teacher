# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os.path as osp
from typing import List
import json

from mmengine.dataset import ConcatDataset
from mmrotate.datasets import DOTADataset
from mmrotate.registry import DATASETS


@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([DATASETS.build(sup), DATASETS.build(unsup)], **kwargs)

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]


@DATASETS.register_module()
class SemiDOTADataset(DOTADataset):
    """DOTA-v1.0 dataset for detection.

    Note: ``ann_file`` in DOTADataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTADataset,
    it is the path of a folder containing XML files.

    Args:
        diff_thr (int): The difficulty threshold of ground truth. Bboxes
            with difficulty higher than it will be ignored. The range of this
            value should be non-negative integer. Defaults to 100.
        img_suffix (str): The suffix of images. Defaults to 'png'.
    """

    def __init__(self,
                 division_file=None,
                 supervised=True,
                 fold=1,
                 percentage=10.0,
                 img_shape=(1024, 1024),
                 diff_thr=100,
                 **kwargs) -> None:
        self.diff_thr = diff_thr
        self.img_shape = img_shape

        self.division_file = division_file
        self.supervised = supervised
        self.fold = fold
        self.percentage = float(percentage)

        super().__init__(**kwargs)
    
    def __load_data_with_annfiles(self, txt_files):
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        for txt_file in txt_files:
            data_info = {}
            img_id = osp.split(txt_file)[1][:-4]
            data_info['img_id'] = img_id
            img_name = img_id + '.png'
            data_info['file_name'] = img_name
            data_info['img_path'] = osp.join(self.data_prefix['img_path'],
                                             img_name)
            data_info['height'] = self.img_shape[0]
            data_info['width'] = self.img_shape[1]

            instances = []
            with open(txt_file) as f:
                s = f.readlines()
                for si in s:
                    instance = {}
                    bbox_info = si.split()
                    instance['bbox'] = [float(i) for i in bbox_info[:8]]
                    cls_name = bbox_info[8]
                    instance['bbox_label'] = cls_map[cls_name]
                    difficulty = int(bbox_info[9])
                    if difficulty > self.diff_thr:
                        instance['ignore_flag'] = 1
                    else:
                        instance['ignore_flag'] = 0
                    instances.append(instance)
            data_info['instances'] = instances
            data_list.append(data_info)

        return data_list

    def __load_data_without_annfiles(self, img_files):
        data_list = []
        for img_path in img_files:
            data_info = {}
            data_info['img_path'] = img_path
            img_name = osp.split(img_path)[1]
            data_info['file_name'] = img_name
            img_id = img_name[:-4]
            data_info['img_id'] = img_id
            data_info['height'] = self.img_shape[0]
            data_info['width'] = self.img_shape[1]

            instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
            data_info['instances'] = [instance]
            data_list.append(data_info)

        return data_list

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``
        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        
        if self.division_file:
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            # get total set
            total_set = [osp.split(txt_file)[1][:-4] for txt_file in txt_files]
            # divide the sup set
            with open(self.division_file, 'r') as f:
                division_infos = json.load(f)
            sup_percent_set = division_infos[str(self.percentage)][str(self.fold)]
            if self.supervised:
                txt_files = [
                    osp.join(self.ann_file, x + '.txt') for x in sup_percent_set
                ]
                data_list = self.__load_data_with_annfiles(txt_files)
            else:
                unsup_percent_set = list(set(total_set)-set(sup_percent_set))
                img_files = [
                    osp.join(self.data_prefix['img_path'], x + '.png') for x in unsup_percent_set
                ]
                data_list = self.__load_data_without_annfiles(img_files)
        else:  
            # print(self.data_prefix['img_path'])
            if not self.ann_file:
                img_files = glob.glob(
                    osp.join(self.data_prefix['img_path'], '*.png'))
                data_list = self.__load_data_without_annfiles(img_files)
            else:
                txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
                data_list = self.__load_data_with_annfiles(txt_files)
        return data_list

@DATASETS.register_module()
class SemiDOTAv15Dataset(SemiDOTADataset):
    """DOTA-v1.5 dataset for detection.

    Note: ``ann_file`` in DOTAv15Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv15Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter', 'container-crane'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255), (220, 20, 60)]
    }


@DATASETS.register_module()
class SemiDOTAv2Dataset(SemiDOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
         'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
         'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
         'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport',
         'helipad'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
                    (138, 43, 226), (255, 128, 0), (255, 0, 255),
                    (0, 255, 255), (255, 193, 193), (0, 51, 153),
                    (255, 250, 205), (0, 139, 139), (255, 255, 0),
                    (147, 116, 116), (0, 0, 255), (220, 20, 60), (119, 11, 32),
                    (0, 0, 142)]
    }

@DATASETS.register_module()
class SemiDIORDataset(SemiDOTADataset):
    """DOTA-v2.0 dataset for detection.

    Note: ``ann_file`` in DOTAv2Dataset is different from the BaseDataset.
    In BaseDataset, it is the path of an annotation file. In DOTAv2Dataset,
    it is the path of a folder containing XML files.
    """

    METAINFO = {
        'classes':
        ('airplane', 'airport', 'baseballfield', 'basketballcourt','bridge',
         'chimney', 'Expressway-Service-area', 'Expressway-toll-station', 'dam',
         'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship', 'stadium',
         'storagetank', 'tenniscourt', 'trainstation', 'vehicle', 'windmill'),
        # palette is a list of color tuples, which is used for visualization.
        'palette': [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                    (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                    (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                    (175, 116, 175), (250, 0, 30), (165, 42, 42),
                    (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0),
                    (120, 166, 157)]
    }