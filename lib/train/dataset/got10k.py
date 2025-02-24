import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class Got10k(BaseVideoDataset):
    """ GOT-10k dataset.

    Publication:
        GOT-10k: A Large High-Diversity Benchmark for Generic Object Tracking in the Wild
        Lianghua Huang, Xin Zhao, and Kaiqi Huang
        arXiv:1810.11981, 2018
        https://arxiv.org/pdf/1810.11981.pdf

    Download dataset from http://got-10k.aitestunion.com/downloads
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        # split：可以是 'train'、'val'、'train_full'、'vottrain' 或 'votval'。这里是vottrain和votval
        # 每一个存的都是预定义的seq_ids，votaval是vot验证集预定义的seq_ids
        # seq_ids：视频序列的 ID 列表。明确指定了要使用的数据集中的哪些视频序列(样本)。如果指定了 seq_ids，则忽略 split 参数。就使用自己的seq_ids
        # image_loader：用于读取图像的函数，默认使用 jpeg4py_loader。
        # data_fraction：使用的数据集比例，默认为使用整个数据集。
        root = env_settings().got10k_dir if root is None else root
        super().__init__('GOT10k', root, image_loader)

        # all folders inside the root
        # self.sequence_list：text中存的样本名称
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            # ltr_path：'/home/he/project_code/OSTrack/lib/train/dataset/..'
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_val_split.txt')
            elif split == 'train_full':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_train_full_split.txt')
            elif split == 'vottrain':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_vot_train_split.txt')
            elif split == 'votval':
                file_path = os.path.join(ltr_path, 'data_specs', 'got10k_vot_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            # seq_ids = pandas.read_csv(file_path, header=None, squeeze=True, dtype=np.int64).values.tolist()
            seq_ids = pandas.read_csv(file_path, header=None, dtype=np.int64).squeeze("columns").values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))
        # seq_ids作用是从样本中筛选选中的样本返回self.sequence_list，不是筛选样本里面的帧
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        #data_fraction是从筛选到的样本中取多少比例作为最终的数据集样本
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        # 读取样本数据生成一个字典，键是样本的名称，值是样本文件夹中的meta_info.ini文件中的数据组成的字典
        self.sequence_meta_info = self._load_meta_info()
        # 将每一个样本按照所属于的object_class(样本类别)分类，返回一个字典
        # 键为object_class，值为列表，列表中存的是属于这个类别的样本在sequence_list中的下标
        self.seq_per_class = self._build_seq_per_class()

        # self.class_list为样本的类别的列表
        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()

    def get_name(self):
        return 'got10k'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    # 读取样本数据生成一个字典，键是样本的名称，值是样本里面的具体数据
    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    # 读取样本文件中的meta_info.ini文件，然后读取里面的有用信息，返回一个字典
    # [：-1] 是因为要去掉最后的换行符
    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    # 将每一个样本按照所属于的object_class(样本类别)分类，返回一个字典
    # 键为object_class，值为列表，列表中存的是属于这个类别的样本在sequence_list中的下标
    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        # 读取对应文件夹下的groundtruth.txt文件，将里面的内容存储到gt(ndarray)中
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    # 从absence.label文件中获取全遮挡信息(0,1):0 -- 未遮挡；1 -- 遮挡
    # 从cover.label中获取覆盖信息(0-8)：0 -- 全覆盖；8 -- 没覆盖
    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        # ByteTensor：8位无符号整数 00000000 - 11111111
        # occlusion(ByteTensor)中存的是0，1的遮挡信息
        # cover(ByteTensor)存的是0-8的覆盖信息
        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        # 计算目标在每一帧中的可见性，未遮挡且没有被全覆盖则为1，则可见
        target_visible = ~occlusion & (cover>0).byte()

        # 计算目标在每一帧中的可见比例0 -- 1  1表示完全可见
        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        # bbox存储的是对应文件夹下的groundtruth.txt中的内容，是一个tensor(n,4)   4:x, y, w, h
        bbox = self._read_bb_anno(seq_path)

        # 检查bbox的宽和高是否大于0，大于0则有效
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible, visible_ratio = self._read_target_visible(seq_path)
        # 最终更新可见性，如果可见并且bbox有效则可见
        visible = visible & valid.byte()

        return {'bbox': bbox, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        # frame_list这里已经就将你seq_id(哪个样本(文件夹))中的frame_ids(帧(图片))取出了 list[0].shape = (720,1280,3) (H,W,3)
        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        # anno_frames存放的是所选帧的seq_info_dict信息(bbox,valid,visible,visible_ratio)
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        # frame_list：所选帧的图片
        # anno_frames：所选帧的seq_info_dict信息(bbox,valid,visible,visible_ratio)
        # obj_meta：所选帧属于的样本的meta_info信息(文件夹里面的meta_info.ini的信息)
        return frame_list, anno_frames, obj_meta
