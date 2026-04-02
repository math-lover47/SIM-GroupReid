import json
import math
import cv2
import numpy as np
import os.path as osp
import os
import pickle
from tqdm import tqdm
from collections import defaultdict

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except (ImportError, AttributeError):
    from mediapipe.python.solutions import pose as mp_pose

#Social Interaction Calculate Moudule(SICM)
class PoseProcessor:
    def __init__(self, pt_matrix):
        self.pt_matrix = pt_matrix
        self.pose = mp_pose.Pose(static_image_mode=True, model_complexity=1)

    def extract_keypoints(self, image, bboxes):
        """批量提取所有行人的关键点"""
        keypoints = {}
        for idx, bbox in enumerate(bboxes):
            try:
                x_min, y_min, x_max, y_max = map(int, bbox)
                w = x_max - x_min
                h = y_max - y_min
                cropped = image[y_min:y_max, x_min:x_max]
                if cropped.size == 0:
                    continue

                results = self.pose.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    kps = self._parse_landmarks(results.pose_landmarks.landmark, (w, h))
                    keypoints[idx] = kps
            except Exception as e:
                print(f"Error processing bbox {idx}: {str(e)}")
        return keypoints

    def _parse_landmarks(self, landmarks, img_size):
        """解析关键点坐标"""
        return {
            'left_shoulder': (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * img_size[0],
                              landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * img_size[1]),
            'right_shoulder': (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * img_size[0],
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * img_size[1]),
            'left_elbow': (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * img_size[0],
                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * img_size[1]),
            'left_wrist': (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * img_size[0],
                           landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * img_size[1])
        }


class InteractionCalculator:
    def __init__(self, pt_matrix):
        self.pt_matrix = pt_matrix

    def calculate_pixel_to_meter(self, bbox):
        """根据行人框的宽度计算 pixel_to_meter"""
        x_min, y_min, x_max, y_max = bbox
        w = x_max - x_min
        return 0.6 / w

    def calculate_for_image(self, image, bboxes, keypoints):
        """主计算流程"""
        person_data = []

        # 预处理行人数据
        for idx, bbox in enumerate(bboxes):
            kps = keypoints.get(idx, None)
            position = self._get_position(bbox)
            beta = self._get_beta_vector(kps) if kps else None
            openness = self._get_openness(kps) if kps else 0
            pixel_to_meter = self.calculate_pixel_to_meter(bbox)

            person_data.append({
                'position': position,
                'beta': beta,
                'openness': openness,
                'pixel_to_meter': pixel_to_meter
            })

        # 计算有关键点的人的 pixel_to_meter 平均值
        valid_pixel_to_meter = [p['pixel_to_meter'] for p in person_data if p['beta'] is not None]
        avg_pixel_to_meter = np.mean(valid_pixel_to_meter) if valid_pixel_to_meter else 0.01

        # 计算交互矩阵
        matrix = np.zeros((len(person_data), len(person_data)))
        for i in range(len(person_data)):
            for j in range(i + 1, len(person_data)):
                distance, angle_ab, angle_ba, p = self._calculate_pair(person_data[i], person_data[j],
                                                                       avg_pixel_to_meter)
                matrix[i][j] = p
                matrix[j][i] = p

        return matrix

    def _get_position(self, bbox):
        """计算行人中心位置（透视变换后）"""
        x_min, y_min, x_max, y_max = bbox
        center = ((x_min + x_max) / 2, y_max)  # 使用脚部位置作为基准点
        return self._perspective_transform(center)

    def _perspective_transform(self, point):
        """执行透视变换"""
        p = np.array([point[0], point[1], 1.0])
        transformed = self.pt_matrix @ p
        return (transformed[0] / transformed[2], transformed[1] / transformed[2])

    def _get_beta_vector(self, kps):
        """计算朝向向量"""
        if not kps:
            return None

        ls = kps['left_shoulder']
        rs = kps['right_shoulder']

        # 转换为透视变换后坐标
        ls_tf = self._perspective_transform(ls)
        rs_tf = self._perspective_transform(rs)

        # 计算肩膀向量并旋转90度
        shoulder_vec = np.array(rs_tf) - np.array(ls_tf)
        beta = np.array([-shoulder_vec[1], shoulder_vec[0]])
        return beta / np.linalg.norm(beta) if np.linalg.norm(beta) > 0 else None

    def _get_openness(self, kps):
        """计算姿势开放性"""
        if not kps:
            return 0

        shoulder = np.array(kps['left_shoulder'])
        elbow = np.array(kps['left_elbow'])
        wrist = np.array(kps['left_wrist'])

        upper_arm = elbow - shoulder
        body_vec = np.array([0, 1])

        angle = np.degrees(np.arccos(np.dot(upper_arm, body_vec) /
                                     (np.linalg.norm(upper_arm) * np.linalg.norm(body_vec))))

        if angle > 45:
            return 1
        elif wrist[0] < shoulder[0]:
            return -1
        return 0

    def _calculate_pair(self, a, b, avg_pixel_to_meter):
        """计算两人之间的交互概率"""
        # 计算实际距离
        dx = (a['position'][0] - b['position'][0]) * avg_pixel_to_meter
        dy = (a['position'][1] - b['position'][1]) * avg_pixel_to_meter
        distance = math.sqrt(dx ** 2 + dy ** 2)

        # 检查是否有人没有关键点
        if a['beta'] is None or b['beta'] is None:
            angle_ab = 130  # 如果有人没有关键点，夹角设置为130度
            angle_ba = 130
        else:
            # 计算相对角度
            alpha_ab = np.array(b['position']) - np.array(a['position'])
            alpha_ab = alpha_ab / np.linalg.norm(alpha_ab)
            angle_ab = math.degrees(math.acos(np.dot(alpha_ab, a['beta'])))

            alpha_ba = -alpha_ab
            angle_ba = math.degrees(math.acos(np.dot(alpha_ba, b['beta'])))

        # 计算交互概率
        p = self._probability_formula(distance, angle_ab, angle_ba,
                                      a['openness'], b['openness'])

        return distance, angle_ab, angle_ba, p

    def _probability_formula(self, d, angle_a, angle_b, o_a, o_b):
        c = 0.087
        b = 0.748
        openness_term = (2 * 0.136 * o_a + 1) * (2 * 0.136 * o_b + 1)
        angle_term = (max(math.cos(math.radians(angle_a) / 2), 0) + c) * (
                    max(math.cos(math.radians(angle_b) / 2), 0) + c)
        return 1 - math.exp(-((openness_term * angle_term) / (0.172 * d ** 2)) ** b)


def load_dukegroup_data(annotation_dir):
    """加载DukeGroup数据集并转换为统一格式"""
    # 加载三个JSON文件
    with open(osp.join(annotation_dir, 'group_id.json'), 'r') as f:
        group_data = json.load(f)

    with open(osp.join(annotation_dir, 'person_bounding_box.json'), 'r') as f:
        bbox_data = json.load(f)

    with open(osp.join(annotation_dir, 'person_correspondance.json'), 'r') as f:
        correspondance_data = json.load(f)

    # 创建映射关系
    img_to_bboxes = {item['image name']: item['pedestrian'] for item in bbox_data}
    img_to_group = {}
    group_to_imgs = defaultdict(list)

    for group in group_data:
        group_id = group['id']
        for img_name in group['image names']:
            img_to_group[img_name] = group_id
            group_to_imgs[group_id].append(img_name)

    # 创建person id映射表
    global_pid = 1
    img_pid_to_global = defaultdict(dict)

    # 处理person_correspondance.json建立全局person id映射
    for correspondance in correspondance_data:
        group_pair = correspondance.get('group_pair', [])
        person_pairs = correspondance.get('person pairs', [])

        # 跳过无效的group_pair
        if len(group_pair) < 2:
            print(f"Warning: Invalid group_pair {group_pair}, skipping")
            continue

        img1, img2 = group_pair[0], group_pair[1]

        for pair in person_pairs:
            pid1 = pair.get('person1 id')
            pid2 = pair.get('person2 id')

            # 跳过无效的person pair
            if pid1 is None or pid2 is None:
                continue

            # 如果第一个图像中的person id还没有全局id，则分配一个
            if pid1 not in img_pid_to_global[img1]:
                img_pid_to_global[img1][pid1] = global_pid
                if pid2 not in img_pid_to_global[img2]:
                    img_pid_to_global[img2][pid2] = global_pid
                global_pid += 1
            else:
                # 如果第一个图像中的person id已有全局id，确保第二个图像使用相同的id
                if pid2 not in img_pid_to_global[img2]:
                    img_pid_to_global[img2][pid2] = img_pid_to_global[img1][pid1]

    # 为没有出现在correspondance中的行人分配新的全局id
    for img_name in img_to_bboxes.keys():
        pedestrians = img_to_bboxes[img_name]
        for ped in pedestrians:
            local_pid = ped['person id']
            if local_pid not in img_pid_to_global[img_name]:
                img_pid_to_global[img_name][local_pid] = global_pid
                global_pid += 1

    # 构建CSG格式的数据
    img_names, gids, pids, all_bboxes = [], [], [], []
    # 按组处理
    for group_id, group_imgs in group_to_imgs.items():
        for img_name in group_imgs:
            img_names.append(img_name)
            gids.append(group_id)

            # 获取该图像的所有行人
            pedestrians = img_to_bboxes.get(img_name, [])
            # 使用映射表转换person id
            img_pids = [img_pid_to_global[img_name][ped['person id']] for ped in pedestrians]
            # 保持原始bbox格式 [x_min, y_min, x_max, y_max]
            img_bboxes = [ped['bbox'] for ped in pedestrians]

            pids.append(img_pids)
            all_bboxes.append(img_bboxes)

    return img_names, gids, pids, all_bboxes


def process_dukegroup_dataset(data_dir, annotation_dir, pt_matrix_path, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 修改输出路径为输出目录下的文件名
    output_path = osp.join(output_dir, 'DukeGroup_enhanced.pkl')

    # 加载透视变换矩阵
    pt_matrix = np.load(pt_matrix_path)

    # 加载并转换DukeGroupGroup数据
    img_names, gids, pids, all_bboxes = load_dukegroup_data(annotation_dir)

    # 初始化处理器
    pose_processor = PoseProcessor(pt_matrix)
    interaction_calculator = InteractionCalculator(pt_matrix)

    enhanced_labels = []
    for img_name, bboxes in tqdm(zip(img_names, all_bboxes), desc="Processing", total=len(img_names)):
        img_path = osp.join(data_dir, 'images', img_name)
        if not osp.exists(img_path):
            enhanced_labels.append(None)
            continue

        try:
            image = cv2.imread(img_path)
            if image is None:
                print(f"Warning: Could not read image {img_path}")
                enhanced_labels.append(None)
                continue

            keypoints = pose_processor.extract_keypoints(image, bboxes)
            matrix = interaction_calculator.calculate_for_image(image, bboxes, keypoints)
            enhanced_labels.append(matrix)
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            enhanced_labels.append(None)

    # 保存为pkl文件
    try:
        with open(output_path, 'wb') as f:
            pickle.dump((img_names, gids, pids, all_bboxes, enhanced_labels), f)
        print(f"Successfully saved results to {output_path}")
    except PermissionError:
        print(f"Error: Permission denied when trying to write to {output_path}")
        print("Please choose a different output directory with write permissions.")
    except Exception as e:
        print(f"Error saving results: {str(e)}")


if __name__ == "__main__":
    data_dir = 'E:/Desktop/UMSOT-main/datasets/DukeGroup'  # DukeGroup数据集根目录
    annotation_dir = osp.join(data_dir, 'annotations')
    pt_matrix_path = 'E:/Desktop/UMSOT-main/fastreid/data/transforms/pt_matrix.npy'  # 透视变换矩阵路径
    output_dir = 'E:/Desktop/UMSOT-main/datasets\DukeGroup/annotations'

    process_dukegroup_dataset(data_dir, annotation_dir, pt_matrix_path, output_dir)
