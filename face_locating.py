import cv2
import os
import numpy as np

def detect_faces_dnn(image_path, output_folder):
    # 加载 DNN 模型文件（预训练的 Caffe 模型）
    net = cv2.dnn.readNetFromCaffe(
        'models/deploy.prototxt',  # 配置文件
        'models/res10_300x300_ssd_iter_140000_fp16.caffemodel'  # 预训练权重
    )

    # 读取图像
    image = cv2.imread(image_path)
    (h, w) = image.shape[:2]

    # 预处理图像以适应模型输入
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    net.setInput(blob)
    detections = net.forward()

    # 绘制人脸矩形框
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 如果置信度大于 50%
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)  # 绘制绿色矩形框

    # 保存标记了人脸的图像
    filename = os.path.basename(image_path)
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, image)
    print(f"保存标记人脸的图像：{output_path}")


def detect_faces_in_frames_dnn(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg"):  # 如果是图像文件
            image_path = os.path.join(input_folder, filename)
            detect_faces_dnn(image_path, output_folder)


# 示例：检测 input_frames 文件夹中的每一帧人脸，并将结果保存到 output_faces 文件夹
input_folder = 'extracted_frames'
output_folder = 'output_faces'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
detect_faces_in_frames_dnn(input_folder, output_folder)

#***************************************Histogram of Oriented Gradients算法**************************************************
# import cv2
# import os
#
#
# def detect_faces_hog(image_path, output_folder):
#     # 载入 HOG 人脸检测器
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())  # 使用行人检测器（此处未专门为人脸）
#
#     # 读取图像
#     image = cv2.imread(image_path)
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图像
#
#     # 检测人脸
#     faces, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
#
#     # 如果检测到人脸，绘制矩形框
#     for (x, y, w, h) in faces:
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 绘制绿色矩形框
#
#     # 保存标记了人脸的图像
#     filename = os.path.basename(image_path)
#     output_path = os.path.join(output_folder, filename)
#     cv2.imwrite(output_path, image)
#     print(f"保存标记人脸的图像：{output_path}")
#
#
# def detect_faces_in_frames_hog(input_folder, output_folder):
#     # 遍历文件夹中的所有图像文件
#     for filename in os.listdir(input_folder):
#         if filename.endswith(".png") or filename.endswith(".jpg"):  # 如果是图像文件
#             image_path = os.path.join(input_folder, filename)
#             detect_faces_hog(image_path, output_folder)
#
#
# # 示例：检测输入文件夹中的每一帧人脸，并保存到输出文件夹
# input_folder = 'extracted_frames'  # 输入文件夹（确保文件夹存在）
# output_folder = 'output_faces'  # 输出文件夹
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# detect_faces_in_frames_hog(input_folder, output_folder)
