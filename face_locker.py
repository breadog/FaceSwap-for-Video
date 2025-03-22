from collections import deque

import cv2
import numpy as np
from face_68landmarks import face_68_landmarks
from face_enhancer import enhance_face
from face_recognizer import face_recognize
from face_swap import swap_face
from yolov8face import YOLOface_8n


def initialize_models():
    """初始化所有AI模型"""
    return {
        'detector': YOLOface_8n("weights/yoloface_8n.onnx"),
        'landmarker': face_68_landmarks("weights/2dfan4.onnx"),
        'encoder': face_recognize('weights/arcface_w600k_r50.onnx'),
        'swapper': swap_face('weights/inswapper_128.onnx'),
        'enhancer': enhance_face('weights/gfpgan_1.4.onnx')
    }


MODELS = initialize_models()

class FaceLocker:
    """人脸锁定与追踪系统"""

    def __init__(self, target_embedding):
        self.target_embedding = target_embedding  # 目标人脸特征
        self.tracker = None  # OpenCV跟踪器
        self.landmark_history = deque(maxlen=5)  # 关键点历史记录
        self.last_valid_box = None  # 最后有效区域
        self.failure_count = 0  # 连续失败计数
        self.tracking = False  # 当前追踪状态

    def _box_to_rect(self, box):
        """坐标转换：xyxy -> xywh"""
        return (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))

    def _match_target(self, frame, boxes):
        """在检测到的人脸中匹配目标特征"""
        best_match = None
        max_similarity = 0

        for box in boxes:
            try:
                # 提取人脸特征
                _, landmark = MODELS['landmarker'].detect(frame, box)
                _, embedding = MODELS['encoder'].detect(frame, landmark)

                # 计算相似度
                similarity = np.dot(self.target_embedding, embedding)
                if similarity > max_similarity and similarity > 0.6:  # 相似度阈值
                    max_similarity = similarity
                    best_match = box
            except Exception as e:
                continue

        return best_match, max_similarity

    def update(self, frame):
        """核心更新逻辑"""
        # 阶段1：尝试跟踪现有目标
        if self.tracking:
            success, bbox = self.tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                current_box = np.array([x, y, x + w, y + h], dtype=np.float64)

                # 验证追踪结果
                _, similarity = self._match_target(frame, [current_box])
                if similarity > 0.6:
                    self.failure_count = 0
                    return self._get_landmark(frame, current_box)

            self.failure_count += 1

            # 连续失败3次则重置
            if self.failure_count >= 3:
                self.tracking = False

        # 阶段2：全局重新检测
        boxes, _, _ = MODELS['detector'].detect(frame)
        if not boxes:
            return None

        best_box, similarity = self._match_target(frame, boxes)
        if best_box is not None:
            # 初始化追踪器
            self.tracker = cv2.TrackerKCF_create()
            self.tracker.init(frame, self._box_to_rect(best_box))
            self.tracking = True
            self.failure_count = 0
            return self._get_landmark(frame, best_box)

        return None

    def _get_landmark(self, frame, box):
        """获取平滑后的关键点"""
        _, landmark = MODELS['landmarker'].detect(frame, box)
        self.landmark_history.append(landmark)
        return np.mean(self.landmark_history, axis=0)

