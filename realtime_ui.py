import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QFileDialog, QPushButton
)
from PyQt5.QtCore import Qt, QTimer, QMutex, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QDragEnterEvent, QDropEvent

from realtime_face_swap import prepare_source_face, MODELS


class DragDropLabel(QLabel):
    """支持文件拖拽的标签组件（来自ui.py）"""
    fileDropped = pyqtSignal(str)

    def __init__(self, text, file_types):
        super().__init__()
        self.file_types = file_types
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText(text)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                padding: 20px;
                background-color: #f9f9f9;
            }
        """)
        self.setMinimumSize(200, 150)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if any(url.toLocalFile().lower().endswith(self.file_types) for url in urls):
                event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(self.file_types):
                self.fileDropped.emit(file_path)
                break


class CameraWidget(QLabel):
    """摄像头显示组件（自动启动）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.cap = cv2.VideoCapture(0)
        self.mutex = QMutex()

        if not self.cap.isOpened():
            self.setText("无法打开摄像头")
            self.cap = None
        else:
            self.setText("摄像头初始化中...")

    def get_frame(self):
        """获取当前帧"""
        self.mutex.lock()
        try:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                return frame if ret else None
            return None
        finally:
            self.mutex.unlock()

    def release(self):
        """释放资源"""
        self.mutex.lock()
        try:
            if self.cap:
                self.cap.release()
        finally:
            self.mutex.unlock()


class RealtimeFaceSwapWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("实时换脸")
        self.source_embedding = None
        self.setup_ui()
        self.last_valid_box = None
        self.frame_count = 0
        self.detect_interval = 5  # 每5帧检测一次人脸
        self.file_paths = {
            "source": None
        }
        
        # 设置固定窗口大小
        self.setFixedSize(1280, 900)

        # 自动启动处理定时器
        if self.camera_view.cap:
            self.timer.start(30)  # ~33fps

    def setup_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # ======== 拖拽区域 ========
        self.drop_area = DragDropLabel("拖拽源人脸图片至此\n(.jpg, .png)", (".jpg", ".png"))
        self.drop_area.fileDropped.connect(self.load_source_image)
        main_layout.addWidget(self.drop_area)

        # ======== 摄像头显示区域 ========
        self.camera_view = CameraWidget()
        self.camera_view.setMinimumSize(640, 580)
        main_layout.addWidget(self.camera_view, stretch=1)  # 添加拉伸因子

        # ======== 底部按钮容器 ========
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_container.setLayout(bottom_layout)

        # 切换到视频模式按钮
        self.btn_video_mode = QPushButton("切换到视频处理")
        self.btn_video_mode.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.btn_video_mode.clicked.connect(self.switch_to_video_mode)

        # 切换到图片模式按钮
        self.btn_image_mode = QPushButton("切换到图片处理")
        self.btn_image_mode.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.btn_image_mode.clicked.connect(self.switch_to_image_mode)

        # 添加按钮到底部布局
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(self.btn_video_mode)
        bottom_layout.addWidget(self.btn_image_mode)
        bottom_layout.addStretch(1)

        # ======== 状态栏 ========
        self.status_label = QLabel("就绪 | 等待源人脸图片")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")

        # 组装主布局
        main_layout.addStretch(1)  # 添加弹性空间推动内容向上
        main_layout.addWidget(bottom_container)
        main_layout.addWidget(self.status_label)

        # 处理定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_frame)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def switch_to_video_mode(self):
        """切换到视频处理模式"""
        from ui import MainWindow  # 延迟导入避免循环依赖
        self.close()
        video_window = MainWindow()
        video_window.show()

    def switch_to_image_mode(self):
        """切换到图片处理模式"""
        from image_ui import ImageSwapWindow  # 延迟导入避免循环依赖
        self.close()
        image_window = ImageSwapWindow()
        image_window.show()

    # ... 保持 load_source_image、process_frame 等方法不变 ...

    def closeEvent(self, event):
        """窗口关闭时释放资源"""
        self.timer.stop()
        self.camera_view.release()
        cv2.destroyAllWindows()
        event.accept()

    def load_source_image(self, file_path):
        """加载源图片"""
        try:
            self.source_embedding = prepare_source_face(file_path)
            self.status_label.setText("源图片已加载，开始实时换脸")
            # 显示缩略图
            pixmap = QPixmap(file_path).scaled(100, 100, Qt.KeepAspectRatio)
            self.drop_area.setPixmap(pixmap)
        except Exception as e:
            self.status_label.setText(f"错误: {str(e)}")
            self.source_embedding = None

    # realtime_ui.py 关键修改部分

    def process_frame(self):
        """实时处理帧"""
        if self.source_embedding is None:
            return

        frame = self.camera_view.get_frame()
        if frame is None:
            return

        try:
            # 分辨率调整
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))

            # 定期检测人脸
            if self.frame_count % self.detect_interval == 0 or self.last_valid_box is None:
                target_boxes, _, _ = MODELS['detector'].detect(frame)

                # 确保 target_boxes 存在且非空
                if target_boxes is not None and len(target_boxes) > 0:
                    # 选择面积最大的边界框
                    main_box = max(target_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                    self.last_valid_box = main_box.astype(np.int32)  # 确保数据类型正确
                else:
                    self.last_valid_box = None

            # 执行换脸（显式检查是否为 None）
            if self.last_valid_box is not None:
                try:
                    # 提取关键点前验证边界框有效性
                    x1, y1, x2, y2 = self.last_valid_box
                    if x1 >= x2 or y1 >= y2 or x2 > frame.shape[1] or y2 > frame.shape[0]:
                        raise ValueError("无效的边界框坐标")

                    # 提取关键点
                    _, target_landmark = MODELS['landmarker'].detect(frame, self.last_valid_box)

                    # 换脸处理
                    swapped_frame = MODELS['swapper'].process(frame, self.source_embedding, target_landmark)
                    swapped_frame = MODELS['enhancer'].process(swapped_frame, target_landmark)
                    self.show_frame(swapped_frame)
                except Exception as e:
                    print(f"处理异常: {str(e)}")
                    self.last_valid_box = None  # 重置检测
                    self.show_frame(frame)  # 显示原始帧
            else:
                self.show_frame(frame)

            self.frame_count += 1

        except Exception as e:
            print(f"全局处理异常: {str(e)}")
            self.last_valid_box = None

    def show_frame(self, frame):
        """显示处理后的帧"""
        try:
            # 转换颜色空间
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb_frame.shape
            bytes_per_line = 3 * w
            q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.camera_view.setPixmap(QPixmap.fromImage(q_img).scaled(
                self.camera_view.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            print(f"显示错误: {str(e)}")

    def closeEvent(self, event):
        """窗口关闭处理"""
        self.camera_view.release()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RealtimeFaceSwapWindow()
    window.show()
    sys.exit(app.exec_())