from PyQt5.QtMultimedia import QMediaPlayer
from PyQt5.QtCore import Qt
import time

class VideoSliderManager:
    @staticmethod
    def seek_frame(window, position):
        """处理视频进度条拖动事件"""
        try:
            # 检查边界
            if position < 0:
                position = 0
            if position >= window.source_video.total_frames:
                position = window.source_video.total_frames - 1

            # 更新当前帧
            window.current_frame = position

            # 获取并显示当前帧
            try:
                src_frame = window.source_video.get_frame(position)
                if src_frame is not None:
                    window.display_frame(window.source_video, src_frame)
            except Exception as e:
                print(f"源视频帧读取错误: {str(e)}")

            # 如果有输出视频，也更新它
            if hasattr(window, 'output_video') and window.output_video.cap is not None:
                try:
                    out_frame = window.output_video.get_frame(position)
                    if out_frame is not None:
                        window.display_frame(window.output_video, out_frame)
                except Exception as e:
                    print(f"输出视频帧读取错误: {str(e)}")

            # 更新进度条和时间显示
            window.slider.setValue(position)
            VideoSliderManager.update_slider_time(window)

            # 同步音频位置（如果有）
            if window.source_video.fps > 0 and window.media_player.state() != QMediaPlayer.StoppedState:
                try:
                    audio_position = int((position / window.source_video.fps) * 1000)
                    window.media_player.setPosition(audio_position)
                except Exception as e:
                    print(f"音频同步错误: {str(e)}")

        except Exception as e:
            print(f"进度条拖动错误: {str(e)}")
            window.show_status("视频定位失败，请重试", error=True)

    @staticmethod
    def update_slider_time(window):
        """更新进度条时间显示"""
        try:
            fps = window.source_video.fps
            if fps <= 0:  # 防止除以零错误
                fps = 30
            total_time = time.strftime("%M:%S", time.gmtime(window.source_video.total_frames / fps))
            current_time = time.strftime("%M:%S", time.gmtime(window.current_frame / fps))
            window.time_label.setText(f"{current_time} / {total_time}")
        except Exception as e:
            print(f"更新时间显示错误: {str(e)}")

    @staticmethod
    def setup_slider(window):
        """设置进度条的基本属性和连接"""
        window.slider.setOrientation(Qt.Horizontal)
        window.slider.setMinimum(0)
        window.slider.sliderMoved.connect(lambda pos: VideoSliderManager.seek_frame(window, pos))
        window.slider.setEnabled(False)  # 初始状态禁用 