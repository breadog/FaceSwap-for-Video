import shutil
import sys
import os
import cv2
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QMutex, QUrl
from PyQt5.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent

import main
import video_processing
from video_processing import extract_frames_and_audio
from refresh_manager import RefreshManager  # 导入刷新管理器
from video_slider import VideoSliderManager  # 导入进度条管理器

# ------------------------- 视频播放组件 -------------------------
class VideoPlayer(QLabel):
    frame_updated = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setText("暂无视频")
        self.setStyleSheet("background-color: black; color: white;")
        self.cap = None
        self.total_frames = 0
        self.current_frame = 0
        self.fps = 30  # 默认帧率
        self.mutex = QMutex()
        self.last_frame_time = 0  # 添加时间戳记录

    def load_video(self, path):
        self.mutex.lock()
        try:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise ValueError(f"无法打开视频文件 {path}")

            # 正确获取视频参数
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            # 处理异常帧率（部分视频可能返回0）
            if self.fps <= 0:
                self.fps = 30  # 设置默认值
            print(f"已加载视频：帧率={self.fps}, 总帧数={self.total_frames}")
            return True
        except Exception as e:
            print(f"视频加载错误: {str(e)}")
            return False
        finally:
            self.mutex.unlock()

    def get_frame(self, frame_num):
        if not self.cap or not self.cap.isOpened():
            return None

        self.mutex.lock()
        try:
            # 检查帧号是否有效
            if frame_num < 0 or frame_num >= self.total_frames:
                return None

            # 设置帧位置
            if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num):
                print(f"无法设置视频帧位置: {frame_num}")
                return None

            # 读取帧
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print(f"无法读取视频帧: {frame_num}")
                return None

            try:
                # 检查颜色通道数
                if len(frame.shape) == 2:  # 如果是单通道（灰度图）
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:  # 如果是RGBA
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                
                # 转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame
            except Exception as e:
                print(f"颜色空间转换错误: {str(e)}")
                return None

        except Exception as e:
            print(f"读取帧错误: {str(e)}")
            return None
        finally:
            self.mutex.unlock()


# ------------------------- 文件拖拽组件 -------------------------
class DragDropLabel(QLabel):
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


# ------------------------- 处理线程 -------------------------
class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int)  # 进度信号
    stage_updated = pyqtSignal(str)  # 阶段更新信号
    processing_finished = pyqtSignal(bool)  # 处理完成信号，带成功/失败状态

    def __init__(self, source, target, video):
        super().__init__()
        self.source = source
        self.target = target
        self.video = video
        self.output = os.path.join(os.getcwd(), "output.mp4")
        self.temp_dir = "temp_frames"
        self.total_frames = 0

    def run(self):
        try:
            # 获取视频信息
            video_info = video_processing.get_video_info(self.video)
            original_fps = video_info['fps']
            self.total_frames = video_info['total_frames']

            # 阶段1: 提取视频帧
            self.stage_updated.emit("正在提取视频帧...")
            extract_frames_and_audio(
                video_path=self.video,
                output_frame_dir=os.path.join('temp_frames'),
                output_audio_path=os.path.join('temp_frames/audio.mp3')
            )

            # 阶段2: 处理视频帧
            self.stage_updated.emit("正在处理视频帧...")

            target_files = [f for f in os.listdir('temp_frames')
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            total_frames = len(target_files)

            for idx, frame_file in enumerate(target_files):
                frame_path = os.path.join('temp_frames', frame_file)
                output_path = os.path.join('temp_frames/processed', f"swapped_{frame_file}")

                # 严格验证帧有效性
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(f"跳过无效帧: {frame_file}")
                    shutil.copy(frame_path, output_path)
                    continue

            main.process_video_frames(
                source_img_path=self.source,
                target_img_path=self.target,
                target_dir=os.path.join('temp_frames'),
                output_dir=os.path.join('temp_frames/processed'),
                original_fps=original_fps,
            )

            progress = int((idx + 1) / total_frames * 100)
            self.progress_updated.emit(progress)

            # 阶段3: 生成视频
            self.stage_updated.emit("正在生成最终视频...")
            main.generate_video(
                input_dir=os.path.join('temp_frames/processed'),
                audio_path=os.path.join('temp_frames/audio.mp3'),
                output_path=os.path.join(os.getcwd(), "output.mp4"),
                original_fps=original_fps,
            )

            self.processing_finished.emit(True)
            self.finished.emit()
        except Exception as e:
            self.stage_updated.emit(f"错误: {str(e)}")
            self.processing_finished.emit(False)
            self.finished.emit()

    def __del__(self):
        if self.isRunning():
            self.terminate()
            self.wait()
        print("线程资源已释放")

# ------------------------- 主窗口 -------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸替换工具")
        self.setup_ui()
        self.file_paths = {
            "source": None,
            "target": None,
            "video": None
        }
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.update_frames)
        self.is_playing = False
        self.current_frame = 0
        
        # 初始化音频播放器
        self.media_player = QMediaPlayer()
        self.media_player.stateChanged.connect(self.on_media_state_changed)
        self.media_player.positionChanged.connect(self.on_media_position_changed)
        self.media_player.durationChanged.connect(self.on_media_duration_changed)

    def setup_ui(self):
        main_widget = QWidget()
        layout = QVBoxLayout()

        # 顶部工具栏
        toolbar_layout = QHBoxLayout()
        
        # 刷新按钮
        self.btn_refresh = QPushButton("⟳ 刷新")
        self.btn_refresh.clicked.connect(self.restart_ui)
        self.btn_refresh.setFixedWidth(80)
        self.btn_refresh.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        
        toolbar_layout.addWidget(self.btn_refresh)
        toolbar_layout.addStretch()  # 添加弹性空间，使刷新按钮靠左

        # 文件拖拽区域
        file_layout = QHBoxLayout()
        self.source_label = DragDropLabel("拖拽源人脸图片\n(.jpg, .png)", (".jpg", ".png"))
        self.target_label = DragDropLabel("拖拽目标人脸图片\n(.jpg, .png)", (".jpg", ".png"))
        self.video_label = DragDropLabel("拖拽源视频\n(.mp4, .mov)", (".mp4", ".mov"))

        self.source_label.fileDropped.connect(lambda f: self.set_file(f, "source"))
        self.target_label.fileDropped.connect(lambda f: self.set_file(f, "target"))
        self.video_label.fileDropped.connect(lambda f: self.set_file(f, "video"))

        file_layout.addWidget(self.source_label)
        file_layout.addWidget(self.target_label)
        file_layout.addWidget(self.video_label)

        # 视频播放区域
        video_layout = QHBoxLayout()
        self.source_video = VideoPlayer()
        self.output_video = VideoPlayer()
        # 设置视频播放器的最小尺寸
        self.source_video.setMinimumSize(600, 400)
        self.output_video.setMinimumSize(600, 400)
        video_layout.addWidget(self.source_video)
        video_layout.addWidget(self.output_video)

        # 控制栏
        control_layout = QHBoxLayout()
        self.btn_play = QPushButton("▶")
        self.btn_play.setFixedWidth(40)
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_play.setEnabled(False)

        self.slider = QSlider(Qt.Horizontal)
        VideoSliderManager.setup_slider(self)  # 使用管理器设置进度条

        self.time_label = QLabel("00:00 / 00:00")
        
        # 添加音量控制
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(50)
        self.volume_slider.valueChanged.connect(self.set_volume)
        self.volume_slider.setFixedWidth(100)

        control_layout.addWidget(self.btn_play)
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.time_label)
        control_layout.addWidget(QLabel("音量:"))
        control_layout.addWidget(self.volume_slider)

        # 进度条和状态显示
        progress_layout = QVBoxLayout()
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setFormat("%p% (%v/%m 帧)")
        
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #666;")
        
        progress_layout.addWidget(self.progress)
        progress_layout.addWidget(self.status_label)

        # 底部按钮区域
        bottom_layout = QHBoxLayout()
        
        # 处理按钮
        self.btn_process = QPushButton("开始处理")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_process.setMinimumHeight(40)
        
        # 转换单图模式按钮
        self.btn_switch_mode = QPushButton("转换单图模式")
        self.btn_switch_mode.clicked.connect(self.switch_to_image_mode)
        self.btn_switch_mode.setMinimumHeight(40)
        
        # 切换到摄像头模式按钮
        self.btn_camera = QPushButton("切换到摄像头模式")
        self.btn_camera.clicked.connect(self.switch_to_camera_mode)
        self.btn_camera.setMinimumHeight(40)
        
        # 设置按钮样式
        button_style = """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """
        self.btn_process.setStyleSheet(button_style)
        self.btn_switch_mode.setStyleSheet(button_style)
        self.btn_camera.setStyleSheet(button_style)
        
        bottom_layout.addWidget(self.btn_process)
        bottom_layout.addWidget(self.btn_switch_mode)
        bottom_layout.addWidget(self.btn_camera)

        # 组装界面
        layout.addLayout(toolbar_layout)  # 添加工具栏
        layout.addLayout(file_layout)
        layout.addLayout(video_layout)
        layout.addLayout(control_layout)
        layout.addLayout(progress_layout)
        layout.addLayout(bottom_layout)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)
        self.resize(1280, 800)

    def set_file(self, file_path, file_type):
        self.file_paths[file_type] = file_path
        if file_type == "video":
            # 先获取视频信息
            video_info = video_processing.get_video_info(file_path)
            if self.source_video.load_video(file_path):
                # 使用video_info中的fps
                self.source_video.fps = video_info['fps']
                self.slider.setMaximum(self.source_video.total_frames)
                self.btn_play.setEnabled(True)
                self.update_slider()
                
                # 设置音频
                self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
                
                # 显示第一帧
                first_frame = self.source_video.get_frame(0)
                if first_frame is not None:
                    self.display_frame(self.source_video, first_frame)
                    # 显示成功提示
                    self.show_status(f"视频加载成功：{os.path.basename(file_path)}")
                else:
                    self.show_status("视频加载失败：无法读取第一帧", error=True)
        else:
            pixmap = QPixmap(file_path)
            if not pixmap.isNull():
                scaled_pix = pixmap.scaled(180, 120, Qt.KeepAspectRatio)
                getattr(self, f"{file_type}_label").setPixmap(scaled_pix)
                # 显示成功提示
                file_type_names = {"source": "源人脸", "target": "目标人脸"}
                self.show_status(f"{file_type_names.get(file_type, '')}图片加载成功：{os.path.basename(file_path)}")
            else:
                self.show_status(f"图片加载失败：{os.path.basename(file_path)}", error=True)

    def start_processing(self):

        temp_dir = "temp_frames"
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)  # 递归删除整个目录
                print(f"已清理临时目录: {temp_dir}")
            os.makedirs(temp_dir, exist_ok=True)  # 重新创建空目录
        except Exception as e:
            self.show_status(f"清理临时文件失败: {str(e)}", error=True)
            return  # 终止处理流程

        if not all(self.file_paths.values()):
            self.show_status("请先拖拽所有必需文件！", error=True)
            return

        self.progress.setVisible(True)
        self.progress.setValue(0)

        #启动换脸后先隐藏按钮
        self.btn_process.setEnabled(False)
        self.btn_camera.setEnabled(False)
        self.btn_switch_mode.setEnabled(False)

        self.worker = ProcessingThread(
            self.file_paths["source"],
            self.file_paths["target"],
            self.file_paths["video"]
        )
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.stage_updated.connect(self.update_status)
        self.worker.processing_finished.connect(self.on_processing_finished)
        self.worker.start()

    def update_progress(self, value):
        self.progress.setValue(value)

    def update_status(self, message):
        self.status_label.setText(message)

    def on_processing_finished(self, success):
        try:
            self.progress.setVisible(False)

            self.btn_process.setEnabled(True)
            self.btn_camera.setEnabled(True)
            self.btn_switch_mode.setEnabled(True)
            
            if success:
                # 确保输出视频文件存在
                output_path = os.path.abspath("output.mp4")
                if not os.path.exists(output_path):
                    raise FileNotFoundError(f"输出视频文件不存在: {output_path}")
                
                # 等待文件写入完成
                time.sleep(1)
                
                # 重置输出视频组件
                if self.output_video.cap is not None:
                    self.output_video.cap.release()
                    self.output_video.cap = None
                
                # 加载新视频
                if not self.output_video.load_video(output_path):
                    raise RuntimeError("输出视频加载失败")
                
                # 同步视频参数
                self.output_video.fps = self.source_video.fps  # 使用相同的帧率
                self.slider.setMaximum(self.source_video.total_frames)
                self.current_frame = 0
                
                # 确保两个视频都准备好
                if self.source_video.cap is None or self.output_video.cap is None:
                    raise RuntimeError("视频组件未正确初始化")
                
                # 显示输出视频的第一帧
                first_frame = self.output_video.get_frame(0)
                if first_frame is not None:
                    self.display_frame(self.output_video, first_frame)
                
                # 更新显示
                self.update_frames()
                self.show_status("处理完成！视频已生成")
                
                # 重置播放状态
                self.is_playing = False
                self.btn_play.setText("▶")
                if self.play_timer.isActive():
                    self.play_timer.stop()
                
                # 确保播放按钮可用
                self.btn_play.setEnabled(True)
                
            else:
                self.show_status("处理失败！", error=True)
                
        except Exception as e:
            print(f"错误: {str(e)}")
            self.show_status(f"错误: {str(e)}", error=True)
            # 确保UI状态正确
            self.progress.setVisible(False)
            self.btn_process.setEnabled(True)
            self.is_playing = False
            self.btn_play.setText("▶")
            if self.play_timer.isActive():
                self.play_timer.stop()

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("⏸" if self.is_playing else "▶")

        if self.is_playing:
            # 根据实际帧率计算间隔（单位：ms）
            actual_fps = self.source_video.fps
            interval = int(1000 / actual_fps)  # 动态计算间隔
            print(f"启动播放器，帧率={actual_fps}，间隔={interval}ms")

            self.last_frame_time = time.time()
            self.play_timer.start(interval)  # 动态设置定时器
            self.media_player.play()  # 播放音频
        else:
            self.play_timer.stop()
            self.media_player.pause()  # 暂停音频

    def update_frames(self):
        try:
            if not self.is_playing:
                return

            # 计算应该推进的帧数
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            frames_to_advance = int(elapsed * self.source_video.fps)

            # 至少推进1帧
            if frames_to_advance < 1:
                return

            # 更新当前帧
            self.current_frame += frames_to_advance
            self.last_frame_time = current_time

            # 检查边界
            if self.current_frame >= self.source_video.total_frames:
                self.current_frame = 0

            # 获取并显示当前帧
            src_frame = self.source_video.get_frame(self.current_frame)
            out_frame = self.output_video.get_frame(self.current_frame)

            if src_frame is not None:
                self.display_frame(self.source_video, src_frame)
            if out_frame is not None:
                self.display_frame(self.output_video, out_frame)

            self.update_slider()

        except Exception as e:
            print(f"更新帧时出错: {str(e)}")
            self.toggle_play()

    def display_frame(self, player, frame):
        try:
            if frame is None:
                return
                
            h, w = frame.shape[:2]
            bytes_per_line = 3 * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(
                player.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            player.setPixmap(pixmap)
        except Exception as e:
            print(f"显示帧时出错: {str(e)}")

    def update_slider(self):
        VideoSliderManager.update_slider_time(self)

    def seek_frame(self, position):
        VideoSliderManager.seek_frame(self, position)

    def set_volume(self, value):
        self.media_player.setVolume(value)

    def on_media_state_changed(self, state):
        if state == QMediaPlayer.StoppedState:
            self.is_playing = False
            self.btn_play.setText("▶")
            self.play_timer.stop()
            self.current_frame = 0
            self.update_frames()

    def on_media_position_changed(self, position):
        # 同步视频帧位置
        if self.source_video.fps > 0:
            frame_position = int((position / 1000) * self.source_video.fps)
            if frame_position != self.current_frame:
                self.current_frame = frame_position
                self.update_frames()

    def on_media_duration_changed(self, duration):
        # 更新进度条范围
        if self.source_video.fps > 0:
            total_frames = int((duration / 1000) * self.source_video.fps)
            self.slider.setMaximum(total_frames)

    def show_status(self, message, error=False):
        self.statusBar().showMessage(message, 5000)
        if error:
            self.statusBar().setStyleSheet("color: red;")
        else:
            self.statusBar().setStyleSheet("")

    def __del__(self):
        try:
            # 停止播放
            if self.play_timer.isActive():
                self.play_timer.stop()
            
            # 停止并释放音频播放器
            self.media_player.stop()
            self.media_player.deleteLater()
            
            # 释放所有视频资源
            if hasattr(self, 'source_video') and self.source_video.cap is not None:
                self.source_video.cap.release()
            if hasattr(self, 'output_video') and self.output_video.cap is not None:
                self.output_video.cap.release()
            print("主窗口资源已释放")
        except Exception as e:
            print(f"释放资源时出错: {str(e)}")

    def switch_to_image_mode(self):
        # TODO: 实现切换到单图模式的功能
        from image_ui import ImageSwapWindow
        self.image_window = ImageSwapWindow()
        self.image_window.show()
        self.close()

    def switch_to_camera_mode(self):
        # TODO: 实现切换到摄像头模式的功能
        from realtime_ui import RealtimeFaceSwapWindow
        self.realtime_window = RealtimeFaceSwapWindow()
        self.realtime_window.show()
        self.close()

    def restart_ui(self):
        """重启UI"""
        RefreshManager.refresh_ui(self)  # 使用刷新管理器的静态方法


if __name__ == "__main__":
    try:
        app = QApplication(sys.argv)
        window = MainWindow()
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        sys.exit(1)