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
# *********单张效果**************
import datetime

# import cv2
# import os
# import subprocess
# import json
# import re
# from tqdm import tqdm  # 进度条工具
# from yolov8face import YOLOface_8n
# from face_68landmarks import face_68_landmarks
# from face_recognizer import face_recognize
# from face_swap import swap_face
# from face_enhancer import enhance_face
#
#
# def batch_face_swap(
#         source_img_path: str,
#         target_dir: str,
#         output_dir: str,
#         min_face_size: int = 100,  # 最小人脸尺寸（像素）
#         enable_enhance: bool = True
# ):
#     """
#     批量人脸替换主函数
#     :param source_img_path: 源人脸图片路径
#     :param target_dir: 目标图片目录
#     :param output_dir: 输出目录
#     :param min_face_size: 过滤过小人脸检测结果
#     :param enable_enhance: 是否启用图像增强
#     """
#     # 初始化模型（全局加载避免重复初始化）
#     detect_face_net = YOLOface_8n("weights/yoloface_8n.onnx")
#     detect_68landmarks_net = face_68_landmarks("weights/2dfan4.onnx")
#     face_embedding_net = face_recognize('weights/arcface_w600k_r50.onnx')
#     swap_face_net = swap_face('weights/inswapper_128.onnx')
#     enhance_face_net = enhance_face('weights/gfpgan_1.4.onnx')
#
#     # 处理源人脸
#     source_img = cv2.imread(source_img_path)
#     source_boxes, _, _ = detect_face_net.detect(source_img)
#     if not source_boxes:
#         raise ValueError("源图片中未检测到人脸！")
#
#     # 取源图中最大的人脸
#     source_box = max(source_boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
#     _, face_landmark_5of68 = detect_68landmarks_net.detect(source_img, source_box)
#     source_embedding, _ = face_embedding_net.detect(source_img, face_landmark_5of68)
#
#     # 创建输出目录
#     os.makedirs(output_dir, exist_ok=True)
#
#     # 遍历目标目录
#     target_files = [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
#     for filename in tqdm(target_files, desc="Processing Images"):
#         target_path = os.path.join(target_dir, filename)
#         output_path = os.path.join(output_dir, f"swapped_{filename}")
#
#         try:
#             # 读取目标图片
#             target_img = cv2.imread(target_path)
#             if target_img is None:
#                 print(f"警告：无法读取 {filename}，跳过")
#                 continue
#
#             # 检测目标人脸
#             target_boxes, _, _ = detect_face_net.detect(target_img)
#             valid_boxes = [
#                 box for box in target_boxes
#                 if (box[2] - box[0]) > min_face_size and (box[3] - box[1]) > min_face_size
#             ]
#
#             if not valid_boxes:
#                 #没有检测到有效人脸也要保存
#                 print(f"警告：{filename} 中未检测到有效人脸")
#                 original_output_path = os.path.join(output_dir, swapped_img)
#                 cv2.imwrite(original_output_path, target_img)
#                 continue
#
#             # 处理每个检测到的人脸（默认处理第一个）
#             main_box = valid_boxes[0]
#             _, target_landmark_5 = detect_68landmarks_net.detect(target_img, main_box)
#
#             # 换脸 + 增强
#             swapped_img = swap_face_net.process(target_img, source_embedding, target_landmark_5)
#             if enable_enhance:
#                 swapped_img = enhance_face_net.process(swapped_img, target_landmark_5)
#
#             # 保存结果
#             cv2.imwrite(output_path, swapped_img)
#
#         except Exception as e:
#             print(f"处理 {filename} 时发生错误：{str(e)}")
#             original_output_path = os.path.join(output_dir, f"swapped_{filename}")
#             cv2.imwrite(original_output_path, target_img)
#             continue
#
# def get_audio_duration(audio_path):
#         """ 获取音频时长（秒） """
#         cmd = [
#             'ffprobe', '-v', 'error',
#             '-show_entries', 'format=duration',
#             '-of', 'json',
#             audio_path
#         ]
#         try:
#             result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#             metadata = json.loads(result.stdout)
#             return float(metadata['format']['duration'])
#         except Exception as e:
#             print(f"无法获取音频时长: {str(e)}")
#             return None
#
# def generate_synced_video(
#         image_folder: str,
#         output_video_path: str,
#         audio_path: str,
#         image_prefix: str = 'swapped_frame_%04d.jpg'
# ):
#     """
#     生成与音频严格同步的视频
#     :param image_folder: 包含处理后的图片目录
#     :param output_video_path: 输出视频路径
#     :param audio_path: 音频文件路径
#     :param image_prefix: 图片命名格式（如swapped_%04d.jpg）
#     """
#     # 获取有效图片列表并按数字排序
#     image_files = sorted(
#         [f for f in os.listdir(image_folder) if f.startswith('swapped_') and f.endswith(('.jpg', '.jpeg', '.png'))],
#         key=lambda x: int(re.search(r'(\d+)', x).group(1))  # 提取文件名中的数字部分排序
#     )
#
#     if len(image_files) == 0:
#         raise ValueError("没有找到处理后的图片文件")
#
#     # 自动修正图片前缀格式
#     sample_name = image_files[0]
#     base_name = os.path.splitext(sample_name)[0]  # 去除扩展名
#     match = re.search(r'(\d+)$', base_name)  # 匹配末尾的数字部分
#     if not match:
#         raise ValueError(f"文件名 {sample_name} 中未找到末尾的数字序列")
#
#     num_str = match.group(1)
#     padding = len(num_str)
#     prefix_part = base_name.rstrip(num_str)  # 获取数字前的固定前缀
#     actual_prefix = f"{prefix_part}%0{padding}d.jpg"  # 构造格式字符串
#     input_pattern = os.path.join(image_folder, actual_prefix)
#
#     # 获取音频时长
#     audio_duration = get_audio_duration(audio_path)
#     if audio_duration is None:
#         return
#
#     # 计算动态帧率
#     image_count = len(image_files)
#     required_fps = image_count / audio_duration
#
#     # 生成视频命令
#     ffmpeg_cmd = [
#         'ffmpeg', '-y',
#         '-r', str(required_fps),
#         '-i', input_pattern,
#         '-i', audio_path,
#         '-c:v', 'libx264',
#         '-c:a', 'aac',
#         '-pix_fmt', 'yuv420p',
#         '-vf', f'fps={required_fps},format=yuv420p',
#         '-map', '0:v:0',
#         '-map', '1:a:0',
#         '-shortest',
#         '-movflags', '+faststart',
#         output_video_path
#     ]
#
#     print(f"生成参数：图片数量={image_count} 音频时长={audio_duration:.2f}s 计算帧率={required_fps:.2f}")
#     print(f"FFmpeg命令: {' '.join(ffmpeg_cmd)}")
#
#     try:
#         subprocess.run(ffmpeg_cmd, check=True, stderr=subprocess.PIPE, text=True)
#         print(f"成功生成同步视频: {output_video_path}")
#     except subprocess.CalledProcessError as e:
#         print(f"视频生成失败: {e.stderr}")
#
#
#
# if __name__ == '__main__':
#     # 配置参数
#     SOURCE_IMAGE = "images/crazy.jpg"  # 源人脸图片
#     TARGET_DIR = "video_frames/"  # 目标图片目录
#     OUTPUT_DIR = "output_results/"  # 输出目录
#     AUDIO_PATH = "audio_output.mp3"
#     FINAL_VIDEO = "final_synced_video.mp4"
#
#     batch_face_swap(
#         source_img_path=SOURCE_IMAGE,
#         target_dir=TARGET_DIR,
#         output_dir=OUTPUT_DIR,
#         min_face_size=100,  # 过滤小于100x100的人脸
#         enable_enhance=True  # 启用图像增强
#     )
#
#     # generate_synced_video(
#     #     image_folder=OUTPUT_DIR,
#     #     output_video_path=FINAL_VIDEO,
#     #     audio_path=AUDIO_PATH,
#     #     image_prefix='swapped_frame_%04d.jpg'
#     # )


import cv2
import os
import subprocess
import re
import argparse
import numpy as np
from ffmpeg import output
from matplotlib import pyplot as plt
from tqdm import tqdm
from collections import deque

from face_locker import FaceLocker
from video_processing import get_video_info, extract_frames_and_audio
from yolov8face import YOLOface_8n
from face_68landmarks import face_68_landmarks
from face_recognizer import face_recognize
from face_swap import swap_face
from face_enhancer import enhance_face


# ------------------------- 全局模型初始化 -------------------------
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


# ------------------------- 目标追踪类 -------------------------
# class FaceLocker:
#     """人脸锁定与追踪系统"""
#
#     def __init__(self, target_embedding):
#         self.target_embedding = target_embedding  # 目标人脸特征
#         self.tracker = None  # OpenCV跟踪器
#         self.landmark_history = deque(maxlen=5)  # 关键点历史记录
#         self.last_valid_box = None  # 最后有效区域
#         self.failure_count = 0  # 连续失败计数
#         self.tracking = False  # 当前追踪状态
#
#     def _box_to_rect(self, box):
#         """坐标转换：xyxy -> xywh"""
#         return (int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1]))
#
#     def _match_target(self, frame, boxes):
#         """在检测到的人脸中匹配目标特征"""
#         best_match = None
#         max_similarity = 0
#
#         for box in boxes:
#             try:
#                 # 提取人脸特征
#                 _, landmark = MODELS['landmarker'].detect(frame, box)
#                 _, embedding = MODELS['encoder'].detect(frame, landmark)
#
#                 # 计算相似度
#                 similarity = np.dot(self.target_embedding, embedding)
#                 if similarity > max_similarity and similarity > 0.6:  # 相似度阈值
#                     max_similarity = similarity
#                     best_match = box
#             except Exception as e:
#                 continue
#
#         return best_match, max_similarity
#
#     def update(self, frame):
#         """核心更新逻辑"""
#         # 阶段1：尝试跟踪现有目标
#         if self.tracking:
#             success, bbox = self.tracker.update(frame)
#             if success:
#                 x, y, w, h = [int(v) for v in bbox]
#                 current_box = np.array([x, y, x + w, y + h], dtype=np.float64)
#
#                 # 验证追踪结果
#                 _, similarity = self._match_target(frame, [current_box])
#                 if similarity > 0.6:
#                     self.failure_count = 0
#                     return self._get_landmark(frame, current_box)
#
#             self.failure_count += 1
#
#             # 连续失败3次则重置
#             if self.failure_count >= 3:
#                 self.tracking = False
#
#         # 阶段2：全局重新检测
#         boxes, _, _ = MODELS['detector'].detect(frame)
#         if not boxes:
#             return None
#
#         best_box, similarity = self._match_target(frame, boxes)
#         if best_box is not None:
#             # 初始化追踪器
#             self.tracker = cv2.TrackerKCF_create()
#             self.tracker.init(frame, self._box_to_rect(best_box))
#             self.tracking = True
#             self.failure_count = 0
#             return self._get_landmark(frame, best_box)
#
#         return None
#
#     def _get_landmark(self, frame, box):
#         """获取平滑后的关键点"""
#         _, landmark = MODELS['landmarker'].detect(frame, box)
#         self.landmark_history.append(landmark)
#         return np.mean(self.landmark_history, axis=0)


# ------------------------- 核心处理流程 -------------------------
def process_video_frames(source_img_path, target_img_path, target_dir, output_dir, original_fps, status_callback=None):
    """处理视频帧的主流程"""
    # 初始化源特征
    source_img = cv2.imread(source_img_path)
    if source_img is None:
        raise ValueError(f"无法读取源图片：{source_img_path}")

    # 提取源人脸特征
    source_boxes, _, _ = MODELS['detector'].detect(source_img)
    if not source_boxes:
        raise ValueError("源图片中未检测到人脸")
    source_box = max(source_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    _, source_landmark = MODELS['landmarker'].detect(source_img, source_box)
    source_embedding, _ = MODELS['encoder'].detect(source_img, source_landmark)

    # 初始化目标特征
    target_img = cv2.imread(target_img_path)
    if target_img is None:
        raise ValueError(f"无法读取目标图片：{target_img_path}")

    target_boxes, _, _ = MODELS['detector'].detect(target_img)
    if not target_boxes:
        raise ValueError("目标图片中未检测到人脸")
    target_box = max(target_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    _, target_landmark = MODELS['landmarker'].detect(target_img, target_box)
    _, target_embedding = MODELS['encoder'].detect(target_img, target_landmark)

    # 初始化锁定器
    locker = FaceLocker(target_embedding)
    os.makedirs(output_dir, exist_ok=True)

    # 处理帧序列
    frame_files = sorted(
        [f for f in os.listdir(target_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        key=lambda x: int(re.findall(r'\d+', x)[-1])
    )

    #进度条
    for frame_file in tqdm(frame_files, desc='Processing Frames'):
        if status_callback:
            status_callback(f"正在处理 {frame_file}")
            frame_path = os.path.join(target_dir, frame_file)
            output_path = os.path.join(output_dir, f"swapped_{frame_file}")
            frame = cv2.imread(frame_path)

        if frame is None:
            print(f"警告：无法读取 {frame_file}，跳过")
            continue

        # 执行目标锁定
        target_landmark = locker.update(frame)

        if target_landmark is not None:
            try:
                # 换脸处理
                swapped_frame = MODELS['swapper'].process(frame, source_embedding, target_landmark)
                # 增强处理
                enhanced_frame = MODELS['enhancer'].process(swapped_frame, target_landmark)
                cv2.imwrite(output_path, enhanced_frame)
            except Exception as e:
                print(f"处理 {frame_file} 时出错：{str(e)}")
                cv2.imwrite(output_path, frame)
        else:
            cv2.imwrite(output_path, frame)


# ------------------------- 视频生成模块 -------------------------
def generate_video(input_dir, audio_path, output_path, original_fps):
    """生成最终视频（优化版）"""
    # 获取所有帧文件
    frame_files = sorted(
        [f for f in os.listdir(input_dir)
         if f.startswith('swapped_frame_') and f.endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(re.findall(r'\d+', x)[-1]))

    # 自动补全缺失帧
    expected_count = len(frame_files)
    actual_count = len([f for f in os.listdir(input_dir) if f.startswith('swapped_frame_')])
    if actual_count < expected_count:
        print(f"警告：缺失{expected_count - actual_count}帧，自动补全...")
        last_frame = cv2.imread(os.path.join(input_dir, frame_files[-1]))
        for i in range(actual_count, expected_count):
            cv2.imwrite(os.path.join(input_dir, f"swapped_frame_{i:05d}.jpg"), last_frame)

    # 构建FFmpeg命令
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(original_fps),
        '-i', os.path.join(input_dir, 'swapped_frame_%04d.jpg'),
        '-i', audio_path,
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '18',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-pix_fmt', 'yuv420p',
        '-r', str(original_fps),  # 保持原始帧率
        '-vsync', 'cfr',  # 恒定帧率模式
        '-shortest',  # 音画时长对齐
        '-movflags', '+faststart',
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
        print(f"视频已生成：{output_path}")
    except subprocess.CalledProcessError as e:
        print(f"生成失败：{e.stderr.decode()}")
        raise


# 单张处理
def process_single_image(source_img, target_img):


    detect_face_net = YOLOface_8n("weights/yoloface_8n.onnx")  # 人脸检测模型
    detect_68landmarks_net = face_68_landmarks("weights/2dfan4.onnx")  # 人脸关键点检测模型
    face_embedding_net = face_recognize('weights/arcface_w600k_r50.onnx')  # 人脸特征编码模型
    swap_face_net = swap_face('weights/inswapper_128.onnx')  # 人脸替换模型
    enhance_face_net = enhance_face('weights/gfpgan_1.4.onnx')  # 人脸增强模型

    output_dir = "single_picture"  # 指定目录

    # 获取当前最大序号
    existing_files = [f for f in os.listdir(output_dir)
                      if re.match(r"result_\d{4}\.jpg", f)]

    if existing_files:
        # 提取所有序号并找到最大值
        max_num = max(int(re.search(r"_(\d{4})\.jpg", f).group(1))
                      for f in existing_files)
        new_num = max_num + 1
    else:
        new_num = 1  # 初始序号从0001开始

    # 生成新文件名（固定4位数字补零）
    new_filename = f"result_{new_num:04d}.jpg"
    output_path = os.path.join(output_dir, new_filename)

    #### 处理源图像 ####
    boxes, _, _ = detect_face_net.detect(source_img)  # 检测源图像中人脸的位置
    position = 0  # 假设只处理检测到的第一个人脸
    bounding_box = boxes[position]  # 获取第一个边界框
    _, face_landmark_5of68 = detect_68landmarks_net.detect(source_img, bounding_box)  # 检测关键点
    source_face_embedding, _ = face_embedding_net.detect(source_img, face_landmark_5of68)  # 生成特征向量

    #### 处理目标图像 ####
    boxes, _, _ = detect_face_net.detect(target_img)
    position = 0  ###一张图片里可能有多个人脸，这里只考虑1个人脸的情况
    bounding_box = boxes[position]
    _, target_landmark_5 = detect_68landmarks_net.detect(target_img, bounding_box)

    swapimg = swap_face_net.process(target_img, source_face_embedding, target_landmark_5)  # 替换人脸
    resultimg = enhance_face_net.process(swapimg, target_landmark_5)  # 增强结果


    plt.subplot(1, 2, 1)
    plt.imshow(source_img[:, :, ::-1])  ###plt库显示图像是RGB顺序
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(target_img[:, :, ::-1])
    plt.axis('off')
    # plt.show()
    # plt.savefig('source_target.jpg', dpi=600, bbox_inches='tight')  ###保存高清图


    cv2.imwrite(output_path, resultimg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='高精度人脸替换系统')
    parser.add_argument('--source', required=True, help='源人脸图片路径')
    parser.add_argument('--target', required=True, help='目标锁定图片路径')
    parser.add_argument('--video', required=True, help='原始视频路径')
    parser.add_argument('--output_video', required=True, help='输出视频路径')
    parser.add_argument('--temp_dir', default='temp_frames', help='临时帧目录')

    args = parser.parse_args()

    try:
        # 步骤1：提取原始视频信息
        print("正在分析原始视频...")
        video_info = get_video_info(args.video)
        print(f"视频信息：{video_info['fps']} FPS, 共{video_info['total_frames']}帧")

        # 步骤2：提取视频帧和音频
        print("\n正在提取视频帧和音频...")
        os.makedirs(args.temp_dir, exist_ok=True)
        extract_frames_and_audio(
            video_path=args.video,
            output_frame_dir=args.temp_dir,
            output_audio_path=os.path.join(args.temp_dir, 'audio.mp3')
        )

        # 步骤3：处理视频帧
        print("\n开始人脸替换处理...")
        process_video_frames(
            source_img_path=args.source,
            target_img_path=args.target,
            target_dir=args.temp_dir,
            output_dir=os.path.join(args.temp_dir, 'processed'),
            original_fps=video_info['fps']
        )

        # 步骤4：生成最终视频
        print("\n合成最终视频...")
        generate_video(
            input_dir=os.path.join(args.temp_dir, 'processed'),
            audio_path=os.path.join(args.temp_dir, 'audio.mp3'),
            output_path=args.output_video,
            original_fps=video_info['fps']
        )

    except Exception as e:
        print(f"\n处理失败：{str(e)}")
    finally:
        print("\n清理临时文件...")
        # 可根据需要保留临时文件用于调试
        # shutil.rmtree(args.temp_dir, ignore_errors=True)

    print("处理完成！")
