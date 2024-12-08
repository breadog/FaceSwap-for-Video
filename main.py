import subprocess
import os


def extract_frames_using_ffmpeg(video_path, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 使用 ffmpeg 命令行提取每一帧并保存为图片
    output_pattern = os.path.join(output_folder, 'frame_%04d.png')  # 图片命名规则
    command = [
        'ffmpeg',
        '-i', video_path,  # 输入视频文件
        '-vf', 'fps=10',  # 每秒提取 10 帧 不然合成的时候会卡
        output_pattern  # 输出文件路径
    ]

    try:
        subprocess.run(command, check=True)  # 执行 ffmpeg 命令
        print(f"帧已提取并保存到 {output_folder}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")


# 示例：提取视频帧并保存到 output_frames 文件夹
video_path = 'input/video2.mp4'  # 视频路径
output_folder = 'extracted_frames'  # 输出文件夹
extract_frames_using_ffmpeg(video_path, output_folder)
