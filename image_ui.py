import re
import sys
import os
import time

import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
import main

class DragDropImageLabel(QLabel):
    """可拖拽的图片标签组件"""
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
            QLabel:hover {
                border-color: #666;
                background-color: #f0f0f0;
            }
        """)
        self.setMinimumSize(300, 300)

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

class ProcessingThread(QThread):
    """处理线程"""
    finished = pyqtSignal(bool, str)  # 成功/失败标志，结果图片路径
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)

    def __init__(self, source_path, target_path):
        super().__init__()
        self.source_path = source_path
        self.target_path = target_path
        
        # 确保输出目录存在
        os.makedirs("single_picture", exist_ok=True)
        
        # 生成带时间戳的输出路径
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.output_path = os.path.join("single_picture", f"result_{timestamp}.jpg")

    def find_newest_result(self):
        """查找最新生成的带序号结果文件"""
        result_files = [
            f for f in os.listdir("single_picture")
            if re.match(r"result_\d{4}\.jpg", f)
        ]
        if not result_files:
            return None
        # 按文件名中的序号排序
        sorted_files = sorted(
            result_files,
            key=lambda x: int(re.search(r"_(\d{4})\.jpg", x).group(1)),
            reverse=True
        )
        return os.path.join("single_picture", sorted_files[0])

    def run(self):
        try:
            self.status_updated.emit("正在加载图片...")
            self.progress_updated.emit(20)
            
            # 读取源图片和目标图片
            source_img = cv2.imread(self.source_path)
            if source_img is None:
                raise Exception("无法读取源图片")
                
            target_img = cv2.imread(self.target_path)
            if target_img is None:
                raise Exception("无法读取目标图片")
            
            self.status_updated.emit("正在处理图片...")
            self.progress_updated.emit(50)
            
            # 调用换脸处理函数
            main.process_single_image(source_img, target_img)
            
            # 等待文件写入完成
            time.sleep(0.5)

            result_path = self.find_newest_result()
            if not result_path or not os.path.exists(result_path):
                raise Exception("未找到结果文件，可能生成失败")
            
            # 读取结果图片
            result_img = cv2.imread(result_path)
            if result_img is None:
                raise Exception("无法读取结果图片")
            
            # 保存到指定位置
            if not cv2.imwrite(self.output_path, result_img):
                raise Exception("无法保存结果图片")

            
            self.progress_updated.emit(100)
            self.finished.emit(True, self.output_path)
        except Exception as e:
            print(f"处理出错: {str(e)}")
            self.finished.emit(False, str(e))

class ImageSwapWindow(QMainWindow):
    """单图换脸主窗口"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸替换工具 - 单图模式")
        self.setup_ui()
        self.file_paths = {
            "source": None,
            "target": None
        }


    def setup_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()

        # 标题
        title_label = QLabel("单图人脸替换")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                color: #333;
                margin: 20px;
            }
        """)
        layout.addWidget(title_label)

        # 输入区域
        input_layout = QHBoxLayout()
        
        # 源图片区域
        source_area = QVBoxLayout()
        source_label = QLabel("源人脸图片")
        source_label.setAlignment(Qt.AlignCenter)
        self.source_image = DragDropImageLabel("拖拽源人脸图片\n(.jpg, .png)", (".jpg", ".png"))
        self.source_image.fileDropped.connect(lambda f: self.set_image(f, "source"))
        source_area.addWidget(source_label)
        source_area.addWidget(self.source_image)
        
        # 目标图片区域
        target_area = QVBoxLayout()
        target_label = QLabel("目标图片")
        target_label.setAlignment(Qt.AlignCenter)
        self.target_image = DragDropImageLabel("拖拽目标图片\n(.jpg, .png)", (".jpg", ".png"))
        self.target_image.fileDropped.connect(lambda f: self.set_image(f, "target"))
        target_area.addWidget(target_label)
        target_area.addWidget(self.target_image)

        input_layout.addLayout(source_area)
        input_layout.addLayout(target_area)
        layout.addLayout(input_layout)

        # 输出区域
        output_area = QVBoxLayout()
        output_label = QLabel("处理结果")
        output_label.setAlignment(Qt.AlignCenter)
        self.output_image = QLabel()
        self.output_image.setAlignment(Qt.AlignCenter)
        self.output_image.setStyleSheet("""
            QLabel {
                border: 2px solid #aaa;
                border-radius: 10px;
                background-color: black;
                color: white;
            }
        """)
        self.output_image.setMinimumSize(600, 400)
        self.output_image.setText("处理结果将在这里显示")
        output_area.addWidget(output_label)
        output_area.addWidget(self.output_image)
        layout.addLayout(output_area)

        # 进度条
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #aaa;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        layout.addWidget(self.progress)

        # 状态标签
        self.status_label = QLabel()
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666;")
        layout.addWidget(self.status_label)

        # 按钮区域
        button_layout = QHBoxLayout()
        
        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setMinimumHeight(40)
        self.process_btn.setStyleSheet("""
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
        
        # 切换到视频模式按钮
        self.switch_btn = QPushButton("切换到视频模式")
        self.switch_btn.clicked.connect(self.switch_to_video_mode)
        self.switch_btn.setMinimumHeight(40)
        self.switch_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)

        # 切换到摄像头模式
        self.realtime_btn = QPushButton("实时换脸模式")
        self.realtime_btn.clicked.connect(self.switch_to_realtime_mode)
        self.realtime_btn.setMinimumHeight(40)
        self.realtime_btn.setStyleSheet("""
                    QPushButton {
                        background-color: #FF9800;
                        color: white;
                        border: none;
                        border-radius: 5px;
                        padding: 8px 16px;
                        font-size: 14px;
                        min-width: 120px;
                    }
                    QPushButton:hover { 
                        background-color: #F57C00; 
                    }
                """)


        button_layout.addWidget(self.process_btn)
        button_layout.addWidget(self.switch_btn)
        button_layout.addWidget(self.realtime_btn)
        layout.addLayout(button_layout)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.resize(1280, 800)

    def set_image(self, file_path, image_type):
        """设置图片"""
        try:
            self.file_paths[image_type] = file_path
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                # 缩放图片以适应标签大小
                label = getattr(self, f"{image_type}_image")
                scaled_pixmap = pixmap.scaled(
                    label.size(), 
                    Qt.KeepAspectRatio, 
                    Qt.SmoothTransformation
                )
                label.setPixmap(scaled_pixmap)
                
                # 显示成功消息
                image_type_names = {"source": "源人脸", "target": "目标"}
                self.show_status(f"{image_type_names[image_type]}图片已加载")
            else:
                self.show_status(f"无法加载图片: {file_path}", True)
        except Exception as e:
            self.show_status(f"设置图片时出错: {str(e)}", True)

    def start_processing(self):
        """开始处理图片"""
        if not all(self.file_paths.values()):
            self.show_status("请先拖入源人脸图片和目标图片", True)
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.process_btn.setEnabled(False)
        
        # 创建处理线程
        self.worker = ProcessingThread(
            self.file_paths["source"],
            self.file_paths["target"]
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.status_updated.connect(self.show_status)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.start()

    def update_progress(self, value):
        """更新进度条"""
        self.progress.setValue(value)

    def on_processing_finished(self, success, result):
        """处理完成回调"""
        self.progress.setVisible(False)
        self.process_btn.setEnabled(True)

        if success:
            # 读取并显示结果图片
            result_img = cv2.imread(result)
            if result_img is not None:
                # 将OpenCV的BGR图像转换为RGB
                rgb_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                height, width = rgb_img.shape[:2]
                bytes_per_line = 3 * width
                q_img = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                
                scaled_pixmap = pixmap.scaled(
                    self.output_image.size(),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation
                )
                self.output_image.setPixmap(scaled_pixmap)
                self.show_status("处理完成！")
            else:
                self.show_status("无法加载结果图片", True)
        else:
            self.show_status(f"处理失败: {result}", True)

    def show_status(self, message, is_error=False):
        """显示状态消息"""
        self.status_label.setText(message)
        color = "red" if is_error else "#666"
        self.status_label.setStyleSheet(f"color: {color};")

    def switch_to_video_mode(self):
        """切换到视频模式"""
        from ui import MainWindow
        self.video_window = MainWindow()
        self.video_window.resize(self.size())
        self.video_window.show()
        self.close()

    def switch_to_realtime_mode(self):
        """切换到实时摄像头模式"""
        from realtime_ui import RealtimeFaceSwapWindow  # 延迟导入避免循环依赖

        self.realtime_window = RealtimeFaceSwapWindow()
        self.realtime_window.resize(self.size())
        self.realtime_window.show()
        self.close()


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = ImageSwapWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        sys.exit(1) 