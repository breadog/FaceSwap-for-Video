import cv2
import argparse
import numpy as np
from yolov8face import YOLOface_8n
from face_68landmarks import face_68_landmarks
from face_recognizer import face_recognize
from face_swap import swap_face
from face_enhancer import enhance_face


# ------------------------- 初始化模型 -------------------------
def initialize_models():
    return {
        'detector': YOLOface_8n("weights/yoloface_8n.onnx"),
        'landmarker': face_68_landmarks("weights/2dfan4.onnx"),
        'encoder': face_recognize('weights/arcface_w600k_r50.onnx'),
        'swapper': swap_face('weights/inswapper_128.onnx'),
        'enhancer': enhance_face('weights/gfpgan_1.4.onnx')
    }


MODELS = initialize_models()


# ------------------------- 预处理源人脸 -------------------------
def prepare_source_face(source_img_path):
    source_img = cv2.imread(source_img_path)
    if source_img is None:
        raise ValueError(f"无法读取源图片：{source_img_path}")

    # 检测源人脸
    source_boxes, _, _ = MODELS['detector'].detect(source_img)
    if not source_boxes:
        raise ValueError("源图片中未检测到人脸")

    # 取最大人脸
    source_box = max(source_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    _, source_landmark = MODELS['landmarker'].detect(source_img, source_box)
    source_embedding, _ = MODELS['encoder'].detect(source_img, source_landmark)
    return source_embedding


def realtime_face_swap(source_embedding):
    """实时摄像头人脸替换核心逻辑"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("无法打开摄像头")

    # 配置参数
    window_name = "FaceSwap Live"
    resize_width = 640  # 输入分辨率宽度
    detect_interval = 5  # 全检测间隔帧数
    frame_count = 0
    last_valid_box = None  # 上一次有效人脸框

    # 创建可关闭窗口
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # 检测窗口是否被关闭
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("检测到窗口关闭，退出程序")
                break

            # 读取摄像头帧
            ret, frame = cap.read()
            if not ret:
                print("摄像头帧读取失败")
                break

            # 性能优化：缩小分辨率
            frame = cv2.resize(frame, (resize_width, int(frame.shape[0] * resize_width / frame.shape[1])))
            processed_frame = frame.copy()

            # 定期执行全检测（或首次运行）
            if frame_count % detect_interval == 0 or last_valid_box is None:
                target_boxes, _, _ = MODELS['detector'].detect(frame)
                if target_boxes:
                    # 选择最大人脸
                    main_box = max(target_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    last_valid_box = main_box
                else:
                    last_valid_box = None

            # 处理有效人脸
            if last_valid_box is not None:
                try:
                    # 提取目标关键点
                    _, target_landmark = MODELS['landmarker'].detect(frame, last_valid_box)

                    # 执行换脸和增强
                    swapped_frame = MODELS['swapper'].process(frame, source_embedding, target_landmark)
                    swapped_frame = MODELS['enhancer'].process(swapped_frame, target_landmark)
                    processed_frame = swapped_frame
                except Exception as e:
                    print(f"处理异常: {str(e)}")
                    last_valid_box = None  # 重置检测

            # 显示处理结果
            cv2.imshow(window_name, processed_frame)
            frame_count += 1

            # 键盘退出检测（q或ESC）
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                print("键盘退出指令")
                break

    except Exception as e:
        print(f"程序运行异常: {str(e)}")
    finally:
        # 确保资源释放
        cap.release()
        cv2.destroyAllWindows()
        print("摄像头和窗口资源已释放")
# ------------------------- 主函数 -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, help='源人脸图片路径')
    args = parser.parse_args()

    # 预处理源人脸
    source_embedding = prepare_source_face(args.source)

    # 启动实时处理
    realtime_face_swap(source_embedding)