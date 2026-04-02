import json
import math
import argparse
import cv2
import numpy as np
import os.path as osp
import os
import pickle
from tqdm import tqdm

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
                x, y, w, h = map(int, bbox)
                cropped = image[y:y + h, x:x + w]
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
        x, y, w, h = bbox
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
                distance, angle_ab, angle_ba, p = self._calculate_pair(person_data[i], person_data[j], avg_pixel_to_meter)
                matrix[i][j] = p
                matrix[j][i] = p

        return matrix

    def _get_position(self, bbox):
        """计算行人中心位置（透视变换后）"""
        x, y, w, h = bbox
        center = (x + w / 2, y + h)
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
            angle_ab = 130  # 如果有人没有关键点，夹角设置为90度
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
        openness_term = (2*0.136 * o_a + 1) * (2*0.136 * o_b + 1)
        angle_term = (max(math.cos(math.radians(angle_a) / 2), 0) + c) * (max(math.cos(math.radians(angle_b) / 2), 0) + c)
        return 1 - math.exp(-((openness_term * angle_term) / (0.172 * d ** 2)) ** b)


def process_dataset(data_dir, label_dir, pt_matrix_path):
    # 加载透视变换矩阵
    pt_matrix = np.load(pt_matrix_path)

    # 初始化处理器
    pose_processor = PoseProcessor(pt_matrix)
    interaction_calculator = InteractionCalculator(pt_matrix)

    for split in ['train', 'test', 'gallery']:
        input_path = osp.join(label_dir, f'cuhk_{split}.pkl')
        output_path = osp.join(label_dir, f'cuhk_{split}_enhanced.pkl')

        with open(input_path, 'rb') as f:
            img_names, gids, pids, all_bboxes = pickle.load(f)

        enhanced_labels = []
        for img_name, bboxes in tqdm(zip(img_names, all_bboxes), desc=split):
            img_path = osp.join(data_dir, img_name)
            if not osp.exists(img_path):
                enhanced_labels.append(None)
                continue

            try:
                image = cv2.imread(img_path)
                keypoints = pose_processor.extract_keypoints(image, bboxes)
                matrix = interaction_calculator.calculate_for_image(image, bboxes, keypoints)
                enhanced_labels.append(matrix)
            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")
                enhanced_labels.append(None)

        # 保存增强后的数据
        with open(output_path, 'wb') as f:
            pickle.dump((img_names, gids, pids, all_bboxes, enhanced_labels), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build CSG interaction-enhanced annotations for SIM")
    parser.add_argument("--image-root", required=True, help="Path to CUHK-SYSU group image directory")
    parser.add_argument("--annotation-root", required=True, help="Path to CUHK-SYSU group annotation directory")
    parser.add_argument(
        "--pt-matrix",
        default=osp.join(osp.dirname(__file__), "pt_matrix.npy"),
        help="Path to perspective transform matrix (.npy)",
    )
    args = parser.parse_args()

    process_dataset(args.image_root, args.annotation_root, args.pt_matrix)
