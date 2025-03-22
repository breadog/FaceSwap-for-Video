from PyQt5.QtWidgets import QLabel, QPushButton, QProgressBar
from PyQt5.QtCore import QTimer
from PyQt5.QtMultimedia import QMediaPlayer

class RefreshManager:
    @staticmethod
    def refresh_ui(window):
        """重启UI的所有组件和状态"""
        try:
            # 停止所有正在进行的操作
            if window.play_timer.isActive():
                window.play_timer.stop()
            window.media_player.stop()
            
            # 释放视频资源
            if hasattr(window, 'source_video') and window.source_video.cap is not None:
                window.source_video.cap.release()
            if hasattr(window, 'output_video') and window.output_video.cap is not None:
                window.output_video.cap.release()
            
            # 重置所有状态
            window.file_paths = {
                "source": None,
                "target": None,
                "video": None
            }
            window.current_frame = 0
            window.is_playing = False
            
            # 重置UI元素
            window.source_label.setText("拖拽源人脸图片\n(.jpg, .png)")
            window.target_label.setText("拖拽目标人脸图片\n(.jpg, .png)")
            window.video_label.setText("拖拽源视频\n(.mp4, .mov)")
            window.source_video.setText("暂无视频")
            window.output_video.setText("暂无视频")
            window.btn_play.setText("▶")
            window.btn_play.setEnabled(False)
            window.btn_share.setEnabled(False)
            window.progress.setVisible(False)
            window.progress.setValue(0)
            window.slider.setValue(0)
            window.time_label.setText("00:00 / 00:00")
            
            # 清除图片
            window.source_label.setPixmap(None)
            window.target_label.setPixmap(None)
            
            window.show_status("界面已刷新")
            
        except Exception as e:
            window.show_status(f"刷新失败: {str(e)}", error=True) 