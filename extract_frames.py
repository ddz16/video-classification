import os
import subprocess

def extract_frames(video_path, new_fps, output_path, start_time=None, end_time=None):
    # 创建输出目录
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # 使用FFmpeg获取视频信息
    ffprobe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate,duration', '-of', 'csv=p=0', video_path]
    ffprobe_output = subprocess.check_output(ffprobe_cmd).decode('utf-8').strip()
    width, height, frame_rate, duration = ffprobe_output.split(',')

    # 打印视频信息
    print(f"视频分辨率：{width}x{height}")
    print(f"帧率：{frame_rate}")
    print(f"总时长：{duration}秒")

    # 计算开始和截至时间的时间戳
    duration = int(float(duration))
    start_time = int(start_time) if start_time is not None else 0
    end_time = int(end_time) if end_time is not None else duration-1
    if start_time < 0 or start_time >= duration:
        start_time = 0
    if end_time < start_time or end_time >= duration:
        end_time = duration - 1
    assert start_time < end_time

    # 计算帧间隔
    new_fps = int(new_fps)

    # 使用FFmpeg提取帧
    ffmpeg_cmd = ['ffmpeg', '-i', video_path, '-ss', str(start_time), '-to', str(end_time), '-vf', f"fps={new_fps}", f"{output_path}/%05d.jpg"]
    subprocess.call(ffmpeg_cmd)

    print("帧抽取完成！")


# for video classification
video_path = './data/REC_06-20220910081841/REC_06-20220910081841.mp4'
start_time = 0
end_time = -1
new_fps = 2
output_path = os.path.join(os.path.dirname(video_path), "frames/")

extract_frames(video_path, new_fps, output_path, start_time, end_time)


# for video localization
