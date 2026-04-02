import os.path as osp
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict
from .bases import ImageDataset
from . import DATASET_REGISTRY
import random

# class RoadGroupSplitter:
#     def __init__(self, data_dir, annotation_dir, output_dir):
#         """初始化数据路径"""
#         self.data_dir = data_dir
#         self.annotation_dir = annotation_dir
#         self.output_dir = output_dir
#         os.makedirs(output_dir, exist_ok=True)
#
#     def load_data(self):
#         """加载RoadGroup增强标注数据"""
#         with open(osp.join(self.annotation_dir, 'RoadGroup_enhanced.pkl'), 'rb') as f:
#             img_names, gids, pids, all_bboxes, enhanced_labels = pickle.load(f)
#         return img_names, gids, pids, all_bboxes, enhanced_labels
#
#     def split_data(self, test_size=0.5):
#         """
#         改进的数据划分策略：
#         1. 按组划分训练集(50%)和测试集(50%)，确保组级隔离
#         2. 测试集中每个组随机选1张作为query，其余作为gallery
#         """
#         img_names, gids, pids, all_bboxes, enhanced_labels = self.load_data()
#
#         # 建立组到样本索引的映射字典
#         group_to_indices = defaultdict(list)
#         for idx, gid in enumerate(gids):
#             group_to_indices[gid].append(idx)
#
#         # 获取所有组ID并划分
#         all_groups = list(group_to_indices.keys())
#         train_groups, test_groups = train_test_split(
#             all_groups,
#             test_size=test_size,
#             random_state=42
#         )
#
#         # 初始化各集合索引
#         train_indices = []
#         query_indices = []
#         gallery_indices = []
#
#         # 处理训练集：全量加入
#         for gid in train_groups:
#             train_indices.extend(group_to_indices[gid])
#
#         # 处理测试集：每组随机选1个query，其余gallery
#         for gid in test_groups:
#             group_samples = group_to_indices[gid]
#             random.shuffle(group_samples)  # 随机打乱
#
#             # 确保每组至少有1个query（如果非空组）
#             if group_samples:
#                 query_indices.append(group_samples[0])
#                 gallery_indices.extend(group_samples[1:])
#
#         # 数据分割函数
#         def _split(indices):
#             return (
#                 [img_names[i] for i in indices],
#                 [gids[i] for i in indices],
#                 [pids[i] for i in indices],
#                 [all_bboxes[i] for i in indices],
#                 [enhanced_labels[i] for i in indices] if enhanced_labels else None
#             )
#
#         return _split(train_indices), _split(query_indices), _split(gallery_indices)
#
#     def save_splits(self):
#         """保存划分后的三个子集"""
#         train, query, gallery = self.split_data()
#
#         # 保存训练集
#         with open(osp.join(self.output_dir, 'RoadGroup_train.pkl'), 'wb') as f:
#             pickle.dump(train, f)
#
#         # 保存查询集
#         with open(osp.join(self.output_dir, 'RoadGroup_query.pkl'), 'wb') as f:
#             pickle.dump(query, f)
#
#         # 保存画廊集
#         with open(osp.join(self.output_dir, 'RoadGroup_gallery.pkl'), 'wb') as f:
#             pickle.dump(gallery, f)


@DATASET_REGISTRY.register()
class RoadGroup(ImageDataset):
    _junk_pids = [0, -1]
    dataset_dir = 'RoadGroup'
    dataset_name = "RoadGroup"

    def __init__(self, root='datasets', dataset_root='', image_root='',
                 annotation_root='', enhanced_path='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = dataset_root or osp.join(self.root, self.dataset_dir)
        self.data_dir = image_root or osp.join(self.dataset_dir, 'images')
        self.label_dir = annotation_root or osp.join(self.dataset_dir, 'annotations')
        self.enhanced_path = enhanced_path

        # # 划分数据集（如果尚未划分）
        # splitter = RoadGroupSplitter(self.data_dir, self.label_dir, self.label_dir)
        # if not (osp.exists(osp.join(self.label_dir, 'RoadGroup_train.pkl')))and \
        #         not (osp.exists(osp.join(self.label_dir, 'RoadGroup_query.pkl')))and \
        #              not (osp.exists(osp.join(self.label_dir, 'RoadGroup_gallery.pkl'))):
        #     splitter.save_splits()

        train = self.process_dir(self.data_dir, self.label_dir, 'train')
        query = self.process_dir(self.data_dir, self.label_dir, 'query')
        gallery = self.process_dir(self.data_dir, self.label_dir, 'gallery')

        super(RoadGroup, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, label_path, type):
        if type == 'train':
            labels = osp.join(label_path, f'RoadGroup_{type}.pkl')
            with open(labels, 'rb') as f:
                labels = pickle.load(f)
        elif type == 'query':
            labels = osp.join(label_path, f'RoadGroup_{type}.pkl')
            with open(labels, 'rb') as f:
                labels = pickle.load(f)
        elif type == 'gallery':
            # RoadGroup使用相同的测试集作为gallery
            labels = osp.join(label_path, 'RoadGroup_gallery.pkl')
            with open(labels, 'rb') as f:
                labels = pickle.load(f)

        # elif type == 'gallery':
        #     # RoadGroup使用相同的测试集作为gallery
        #     labels_1 = osp.join(label_path, 'RoadGroup_query.pkl')
        #     with open(labels_1, 'rb') as f:
        #         labels_1 = pickle.load(f)
        #     labels_2 = osp.join(label_path, 'RoadGroup_gallery.pkl')
        #     with open(labels_2, 'rb') as f:
        #         labels_2 = pickle.load(f)
        #
        #     labels = [labels_1[i] + labels_2[i] for i in range(len(labels_1))]

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

            gid = labels[1][idx] + 1  # 组ID从1开始
            pid = labels[2][idx]  # 行人ID
            bbox = labels[3][idx]  # 边界框

            # 设置摄像头ID
            if type == 'train':
                camid = 0
            elif type == 'query':
                camid = 1
            elif type == 'gallery':
                camid = 2

            # 添加数据集前缀
            if type == 'train':
                gid = self.dataset_name + "_" + str(gid)
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)

            data.append((img_path, gid, pid, camid, bbox, avg_probs))
            # data.append((img_path, gid, pid, camid, bbox))

        return data
