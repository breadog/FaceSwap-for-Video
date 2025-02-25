import subprocess
import os


def extract_frames_and_audio(video_path, output_frame_dir, output_audio_path):
    # 创建输出目录
    os.makedirs(output_frame_dir, exist_ok=True)

    # 输出文件名模式
    frame_pattern = os.path.join(output_frame_dir, 'frame_%04d.jpg')

    # 构建FFmpeg命令
    command = [
        'ffmpeg',
        '-i', video_path,  # 输入视频
        '-vf', 'fps=120',  # 视频处理：每秒10帧
        '-q:v', '2',  # JPEG质量 (1-31, 2为最高质量)
        frame_pattern,  # 输出帧路径
        '-vn',  # 禁止处理视频流
        '-y',  # 覆盖已存在文件
        '-acodec', 'libmp3lame',  # 使用MP3编码
        '-q:a', '0',  # 音频质量 0-9 (0为最高)
        output_audio_path  # 输出音频路径
    ]

    try:
        subprocess.run(command, check=True, stderr=subprocess.PIPE, text=True)
        print(f"视频帧已保存到: {output_frame_dir}")
        print(f"音频文件已保存到: {output_audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg错误:\n{e.stderr}")
        raise


if __name__ == '__main__':
    video_path = 'video/output.mp4'  # 输入视频路径
    output_frames = 'video_frames'  # 帧输出目录
    output_audio = 'audio_output.mp3'  # 音频输出路径

    extract_frames_and_audio(video_path, output_frames, output_audio)
