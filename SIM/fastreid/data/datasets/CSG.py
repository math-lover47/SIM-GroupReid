# region
import glob
import pdb
import pickle
import os.path as osp
import re
import warnings
import json
import numpy as np

from methods.SIM.fastreid.data.datasets.bases import ImageDataset
from methods.SIM.fastreid.data.datasets import DATASET_REGISTRY
# from datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class CSG(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = 'CUHK-SYSU'
    dataset_name = "CSG"

    def __init__(self, root='datasets', dataset_root='', image_root='', label_root='',
                 enhanced_root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = dataset_root or osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = image_root or osp.join(self.dataset_dir, 'images')
        self.label_dir = label_root or osp.join(self.dataset_dir, 'GReID_label')
        self.enhanced_root = enhanced_root or self.label_dir

        # self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        # self.query_dir = osp.join(self.data_dir, 'query')
        # self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')

        train = self.process_dir(self.data_dir, self.label_dir, 'train')
        query = self.process_dir(self.data_dir, self.label_dir, 'query')
        gallery = self.process_dir(self.data_dir, self.label_dir, 'gallery')

        super(CSG, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, label_path, type):
        if type == 'train':
            labels = osp.join(self.enhanced_root, f'cuhk_{type}_enhanced.pkl')
            # labels = osp.join(label_path, f'cuhk_{type}.pkl')
            labels = open(labels, 'rb')
            labels = pickle.load(labels)
        elif type == 'query':
            labels = osp.join(self.enhanced_root, f'cuhk_test_enhanced.pkl')
            # labels = osp.join(label_path, f'cuhk_test.pkl')
            labels = open(labels, 'rb')
            labels = pickle.load(labels)
        elif type == 'gallery':
            labels_1 = osp.join(self.enhanced_root, f'cuhk_test_enhanced.pkl')
            # labels_1 = osp.join(label_path, f'cuhk_test.pkl')
            labels_1 = open(labels_1, 'rb')
            labels_1 = pickle.load(labels_1)
            labels_2 = osp.join(self.enhanced_root, f'cuhk_gallery_enhanced.pkl')
            # labels_2 = osp.join(label_path, f'cuhk_gallery.pkl')
            labels_2 = open(labels_2, 'rb')
            labels_2 = pickle.load(labels_2)

            labels = [labels_1[i] + labels_2[i] for i in range(len(labels_1))]
            # pdb.set_trace()

        img_paths = [osp.join(dir_path, x) for x in labels[0]]
        interaction_matrices = labels[4]  # 新增的交互矩阵列表

        data = []
        for idx, img_path in enumerate(img_paths):
            matrix = interaction_matrices[idx]
            avg_probs = None

            # 新增：加载交互概率矩阵
            if matrix is not None:
                interaction_matrix = matrix  # 直接使用 matrix
                avg_probs = np.mean(interaction_matrix, axis=1)

            # in CUHK_SYSU_Group dataset, person and group id start from 0.
            # we force it start from 1
            gid = labels[1][idx] + 1
            pid = labels[2][idx]
            bbox = labels[3][idx]

            camid = -1
            # assert gid >= 0
            # assert 1 <= camid <= 6
            if type == 'train':
                camid = 0
            elif type == 'query':
                camid = 1
            elif type == 'gallery':
                camid = 2

            if type == 'train':
                gid = self.dataset_name + "_" + str(gid)
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
                pid = pid.replace(' ', '')
            data.append((img_path, gid, pid, camid, bbox, avg_probs))

        return data


# endregion
