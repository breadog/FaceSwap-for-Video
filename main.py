# import cv2
# import matplotlib.pyplot as plt
# from yolov8face import YOLOface_8n
# from face_68landmarks import face_68_landmarks
# from face_recognizer import face_recognize
# from face_swap import swap_face
# from face_enhancer import enhance_face
#
# if __name__ == '__main__':
#     source_path = 'images/ldh.jpg'
#     target_path = 'images/lcw.jpg'
#     source_img = cv2.imread(source_path)
#     target_img = cv2.imread(target_path)
#
#     detect_face_net = YOLOface_8n("weights/yoloface_8n.onnx")  # 人脸检测模型
#     detect_68landmarks_net = face_68_landmarks("weights/2dfan4.onnx")  # 人脸关键点检测模型
#     face_embedding_net = face_recognize('weights/arcface_w600k_r50.onnx')  # 人脸特征编码模型
#     swap_face_net = swap_face('weights/inswapper_128.onnx')  # 人脸替换模型
#     enhance_face_net = enhance_face('weights/gfpgan_1.4.onnx')  # 人脸增强模型
#
#     #### 处理源图像 ####
#     boxes, _, _ = detect_face_net.detect(source_img)  # 检测源图像中人脸的位置
#     position = 0  # 假设只处理检测到的第一个人脸
#     bounding_box = boxes[position]  # 获取第一个边界框
#     _, face_landmark_5of68 = detect_68landmarks_net.detect(source_img, bounding_box)  # 检测关键点
#     source_face_embedding, _ = face_embedding_net.detect(source_img, face_landmark_5of68)  # 生成特征向量
#
#     #### 处理目标图像 ####
#     boxes, _, _ = detect_face_net.detect(target_img)
#     position = 0  ###一张图片里可能有多个人脸，这里只考虑1个人脸的情况
#     bounding_box = boxes[position]
#     _, target_landmark_5 = detect_68landmarks_net.detect(target_img, bounding_box)
#
#     swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5)  # 替换人脸
#     resultimg = enhance_face_net.process(swapimg, target_landmark_5)  # 增强结果
#
#     plt.subplot(1, 2, 1)
#     plt.imshow(source_img[:, :, ::-1])  ###plt库显示图像是RGB顺序
#     plt.axis('off')
#     plt.subplot(1, 2, 2)
#     plt.imshow(target_img[:, :, ::-1])
#     plt.axis('off')
#     # plt.show()
#     plt.savefig('source_target.jpg', dpi=600, bbox_inches='tight')  ###保存高清图
#
#     cv2.imwrite('result.jpg', resultimg)

import cv2
import os
from tqdm import tqdm  # 进度条工具
from yolov8face import YOLOface_8n
from face_68landmarks import face_68_landmarks
from face_recognizer import face_recognize
from face_swap import swap_face
from face_enhancer import enhance_face


def batch_face_swap(
        source_img_path: str,
        target_dir: str,
        output_dir: str,
        min_face_size: int = 100,  # 最小人脸尺寸（像素）
        enable_enhance: bool = True
):
    """
    批量人脸替换主函数
    :param source_img_path: 源人脸图片路径
    :param target_dir: 目标图片目录
    :param output_dir: 输出目录
    :param min_face_size: 过滤过小人脸检测结果
    :param enable_enhance: 是否启用图像增强
    """
    # 初始化模型（全局加载避免重复初始化）
    detect_face_net = YOLOface_8n("weights/yoloface_8n.onnx")
    detect_68landmarks_net = face_68_landmarks("weights/2dfan4.onnx")
    face_embedding_net = face_recognize('weights/arcface_w600k_r50.onnx')
    swap_face_net = swap_face('weights/inswapper_128.onnx')
    enhance_face_net = enhance_face('weights/gfpgan_1.4.onnx')

    # 处理源人脸
    source_img = cv2.imread(source_img_path)
    source_boxes, _, _ = detect_face_net.detect(source_img)
    if not source_boxes:
        raise ValueError("源图片中未检测到人脸！")

    # 取源图中最大的人脸
    source_box = max(source_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    _, face_landmark_5of68 = detect_68landmarks_net.detect(source_img, source_box)
    source_embedding, _ = face_embedding_net.detect(source_img, face_landmark_5of68)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 遍历目标目录
    target_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in tqdm(target_files, desc="Processing Images"):
        target_path = os.path.join(target_dir, filename)
        output_path = os.path.join(output_dir, f"swapped_{filename}")

        try:
            # 读取目标图片
            target_img = cv2.imread(target_path)
            if target_img is None:
                print(f"警告：无法读取 {filename}，跳过")
                continue

            # 检测目标人脸
            target_boxes, _, _ = detect_face_net.detect(target_img)
            valid_boxes = [
                box for box in target_boxes
                if (box[2] - box[0]) > min_face_size and (box[3] - box[1]) > min_face_size
            ]

            if not valid_boxes:
                print(f"警告：{filename} 中未检测到有效人脸，跳过")
                continue

            # 处理每个检测到的人脸（默认处理第一个）
            main_box = valid_boxes[0]
            _, target_landmark_5 = detect_68landmarks_net.detect(target_img, main_box)

            # 换脸 + 增强
            swapped_img = swap_face_net.process(target_img, source_embedding, target_landmark_5)
            if enable_enhance:
                swapped_img = enhance_face_net.process(swapped_img, target_landmark_5)

            # 保存结果
            cv2.imwrite(output_path, swapped_img)

        except Exception as e:
            print(f"处理 {filename} 时发生错误：{str(e)}")
            continue


if __name__ == '__main__':
    # 配置参数
    SOURCE_IMAGE = "images/mjr.jpg"  # 源人脸图片
    TARGET_DIR = "video_frames/"  # 目标图片目录
    OUTPUT_DIR = "output_results/"  # 输出目录

    batch_face_swap(
        source_img_path=SOURCE_IMAGE,
        target_dir=TARGET_DIR,
        output_dir=OUTPUT_DIR,
        min_face_size=100,  # 过滤小于100x100的人脸
        enable_enhance=True  # 启用图像增强
    )
