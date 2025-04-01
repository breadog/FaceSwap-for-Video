# from PyQt5.QtMultimedia import QMediaPlayer
# from PyQt5.QtCore import Qt
# import time
#
# class VideoSliderManager:
#     @staticmethod
#     def seek_frame(window, position):
#         """处理视频进度条拖动事件"""
#         try:
#             # 检查边界
#             if position < 0:
#                 position = 0
#             if position >= window.source_video.total_frames:
#                 position = window.source_video.total_frames - 1
#
#             # 更新当前帧
#             window.current_frame = position
#
#             # 获取并显示当前帧
#             try:
#                 src_frame = window.source_video.get_frame(position)
#                 if src_frame is not None:
#                     window.display_frame(window.source_video, src_frame)
#             except Exception as e:
#                 print(f"源视频帧读取错误: {str(e)}")
#
#             # 如果有输出视频，也更新它
#             if hasattr(window, 'output_video') and window.output_video.cap is not None:
#                 try:
#                     out_frame = window.output_video.get_frame(position)
#                     if out_frame is not None:
#                         window.display_frame(window.output_video, out_frame)
#                 except Exception as e:
#                     print(f"输出视频帧读取错误: {str(e)}")
#
#             # 更新进度条和时间显示
#             window.slider.setValue(position)
#             VideoSliderManager.update_slider_time(window)
#
#             # 同步音频位置（如果有）
#             if window.source_video.fps > 0 and window.media_player.state() != QMediaPlayer.StoppedState:
#                 try:
#                     audio_position = int((position / window.source_video.fps) * 1000)
#                     window.media_player.setPosition(audio_position)
#                 except Exception as e:
#                     print(f"音频同步错误: {str(e)}")
#
#         except Exception as e:
#             print(f"进度条拖动错误: {str(e)}")
#             window.show_status("视频定位失败，请重试", error=True)
#
#     @staticmethod
#     def update_slider_time(window):
#         """更新进度条时间显示"""
#         try:
#             fps = window.source_video.fps
#             if fps <= 0:  # 防止除以零错误
#                 fps = 30
#             total_time = time.strftime("%M:%S", time.gmtime(window.source_video.total_frames / fps))
#             current_time = time.strftime("%M:%S", time.gmtime(window.current_frame / fps))
#             window.time_label.setText(f"{current_time} / {total_time}")
#         except Exception as e:
#             print(f"更新时间显示错误: {str(e)}")
#
#     @staticmethod
#     def setup_slider(window):
#         """设置进度条的基本属性和连接"""
#         window.slider.setOrientation(Qt.Horizontal)
#         window.slider.setMinimum(0)
#         window.slider.sliderMoved.connect(lambda pos: VideoSliderManager.seek_frame(window, pos))
#         window.slider.setEnabled(True)  # 初始状态禁用

from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtCore import Qt
import time


class VideoSliderManager:
    @staticmethod
    def setup_slider(window):
        window.slider.sliderMoved.connect(lambda pos: VideoSliderManager.seek_frame(window, pos))
        # 修改信号连接，标记拖动状态
        window.slider.sliderPressed.connect(lambda: setattr(window, 'is_slider_dragging', True))
        window.slider.sliderReleased.connect(lambda: setattr(window, 'is_slider_dragging', False))

    def seek_frame(window, position):
        """处理视频进度条拖动事件（优化版）"""

        # 强化空值检查
        if (
                not hasattr(window, 'source_video') or
                not window.source_video.cap or
                not window.source_video.cap.isOpened()
        ):
            print("错误：视频未加载或已关闭")
            return

        if not window.source_video.cap or not window.source_video.cap.isOpened():
            return

        try:
            # 暂停播放状态
            was_playing = window.is_playing
            if window.is_playing:
                window.toggle_play()

            # 边界检查（新增安全范围限制）
            max_frame = window.source_video.total_frames - 1
            position = max(0, min(position, max_frame))

            window.last_frame_time = time.time()

            # 更新当前帧
            window.current_frame = position

            # 更新显示（优化异常处理）
            try:
                src_frame = window.source_video.get_frame(position)
                window.display_frame(window.source_video, src_frame)

                if window.output_video and window.output_video.cap.isOpened():
                    out_frame = window.output_video.get_frame(position)
                    window.display_frame(window.output_video, out_frame)
            except Exception as e:
                print(f"帧读取错误: {str(e)}")
                return

            # 同步音频（新增有效性检查）
            if (
                    not window.media_player.media().isNull() and  # 检查媒体内容是否为空
                    window.source_video.fps > 0
            ):
                try:
                    audio_position = int((position / window.source_video.fps) * 1000)
                    window.media_player.setPosition(audio_position)
                except Exception as e:
                    print(f"音频同步错误: {str(e)}")

            # 恢复播放状态
            if was_playing:
                window.toggle_play()
                window.toggle_play()  # 重新开

            window.update_frames()

        except Exception as e:
            print(f"进度条操作失败: {str(e)}")
            window.show_status("视频定位失败", error=True)

    @staticmethod
    def update_slider_time(window):
        """精确到毫秒的时间同步"""
        try:
            fps = window.source_video.fps
            if fps <= 0:
                fps = 30  # 默认值

            total_sec = window.source_video.total_frames / fps
            current_sec = window.current_frame / fps

            # 格式化时间为 MM:SS.ms
            total_time = f"{int(total_sec // 60):02d}:{int(total_sec % 60):02d}.{int((total_sec % 1) * 1000):03d}"
            current_time = f"{int(current_sec // 60):02d}:{int(current_sec % 60):02d}.{int((current_sec % 1) * 1000):03d}"

            window.time_label.setText(f"{current_time} / {total_time}")

            window.slider.setValue(window.current_frame)

            window.time_label.setText(f"{current_time} / {total_time}")

            # 强制刷新UI
            window.time_label.repaint()
            window.slider.repaint()

        except Exception as e:
            print(f"时间同步异常: {str(e)}")

    @staticmethod
    def setup_slider(window):
        """完整滑块初始化"""
        window.slider.setOrientation(Qt.Horizontal)
        window.slider.setMinimum(0)

        # 连接所有必要信号
        window.slider.sliderMoved.connect(lambda pos: VideoSliderManager.seek_frame(window, pos))
        window.slider.sliderPressed.connect(window.on_slider_pressed)
        window.slider.sliderReleased.connect(window.on_slider_released)

        # 初始状态设为可用（实际在加载视频时启用）
        window.slider.setEnabled(True)