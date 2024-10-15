kinetics_400_categories = {
    "Sports": [
        "playing basketball", "playing volleyball", "playing tennis", "playing cricket", "playing soccer",
        "swimming", "diving", "surfing", "skiing", "skateboarding", "ice skating", "snowboarding",
        "gymnastics", "pole vault", "long jump", "high jump", "javelin throw", "shot put"
    ],
    "Cooking and Eating": [
        "cooking", "baking", "frying", "boiling", "grilling", "barbecuing", "peeling potatoes",
        "cutting vegetables", "eating", "drinking", "making coffee", "making tea"
    ],
    "Music and Performance": [
        "playing guitar", "playing drums", "playing piano", "playing violin", "singing",
        "dancing", "breakdancing", "tap dancing", "ballet", "conducting orchestra"
    ],
    "Daily Activities": [
        "brushing teeth", "brushing hair", "washing hands", "shaving", "applying makeup",
        "getting dressed", "tying tie", "tying shoe laces", "ironing"
    ],
    "Outdoor Activities": [
        "hiking", "camping", "fishing", "hunting", "rock climbing", "mountain climbing",
        "gardening", "mowing lawn", "planting trees"
    ],
    "Household Chores": [
        "vacuuming", "mopping floor", "washing dishes", "doing laundry", "folding clothes",
        "making bed", "cleaning windows", "dusting"
    ],
    "Work and Professional": [
        "typing", "writing", "reading", "filing nails", "welding", "sewing", "knitting",
        "operating computer", "using atm", "archery"
    ],
    "Entertainment and Games": [
        "playing video games", "playing cards", "playing chess", "playing darts",
        "juggling", "magic tricks", "hula hooping"
    ],
    "Transportation": [
        "driving car", "riding motorcycle", "riding bicycle", "riding bus", "riding train",
        "flying airplane", "sailing", "rowing boat"
    ],
    "Pet and Animal Related": [
        "walking dog", "grooming dog", "grooming horse", "feeding birds", "feeding fish",
        "petting animal"
    ],
    "Beauty and Personal Care": [
        "cutting hair", "dying hair", "getting a haircut", "manicure", "massage",
        "shaving head", "waxing legs"
    ],
    "Social Interaction": [
        "hugging", "kissing", "shaking hands", "high five", "fist bump",
        "celebrating", "cheering", "applauding"
    ]
}

import os
import random
import csv
from moviepy.editor import VideoFileClip, concatenate_videoclips
import json

def load_video_labels(label_file):
    video_labels = {}
    with open(label_file, 'r') as f:
        for line in f:
            filename, label_id = line.strip().split()
            video_labels[filename] = int(label_id)
    return video_labels

def load_label_map(csv_file):
    label_map = {}
    with open(csv_file, 'r') as f:
        csv_reader = csv.DictReader(f)
        for row in csv_reader:
            label_map[int(row['id'])] = row['name']
    return label_map

def create_composite_video(category, n_per_label, video_dir, output_dir, label_file, csv_file):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载视频标签和标签映射
    video_labels = load_video_labels(label_file)
    label_map = load_label_map(csv_file)

    # 获取指定类别的所有标签
    labels = kinetics_400_categories[category]

    # 用于存储每个标签的视频
    label_videos = {label: [] for label in labels}

    # 遍历视频目录，按标签分类视频
    for filename, label_id in video_labels.items():
        if os.path.exists(os.path.join(video_dir, filename)):
            video_label = label_map[label_id]
            if video_label in labels:
                label_videos[video_label].append(os.path.join(video_dir, filename))

    # 打印每个标签找到的视频数量
    for label, videos in label_videos.items():
        print(f"Found {len(videos)} videos for label: {label}")

    # 选择视频并创建剪辑
    clips = []
    video_info = []
    current_time = 0

    # 设置目标分辨率和帧率
    target_resolution = (1280, 720)  # 可以根据需要调整
    target_fps = 30  # 可以根据需要调整

    for label in labels:
        available_videos = label_videos[label]
        if len(available_videos) >= n_per_label:
            selected_videos = random.sample(available_videos, n_per_label)
        else:
            selected_videos = available_videos  # 如果视频不够，就用所有可用的

        print(f"Selected {len(selected_videos)} videos for label: {label}")

        for video_path in selected_videos:
            try:
                clip = VideoFileClip(video_path)
                
                # 调整分辨率
                clip = clip.resize(target_resolution)
                
                # 调整帧率
                clip = clip.set_fps(target_fps)
                
                # 确保音频采样率一致（如果有音频）
                if clip.audio:
                    clip = clip.set_audio(clip.audio.set_fps(44100))
                
                clips.append(clip)
                
                video_info.append({
                    "label": label,
                    "filename": os.path.basename(video_path),
                    "start_time": current_time,
                    "end_time": current_time + clip.duration
                })
                
                current_time += clip.duration
            except Exception as e:
                print(f"Error processing video {video_path}: {str(e)}")

    if not clips:
        print("No valid video clips found. Cannot create composite video.")
        return

    # 拼接视频
    final_clip = concatenate_videoclips(clips, method="compose")

    # 保存最终视频
    output_video_path = os.path.join(output_dir, f"{category}_composite.mp4")
    final_clip.write_videofile(output_video_path, codec='libx264', audio_codec='aac')

    # 保存视频信息
    info_path = os.path.join(output_dir, f"{category}_video_info.json")
    with open(info_path, 'w') as f:
        json.dump(video_info, f, indent=4)

    print(f"Composite video saved to: {output_video_path}")
    print(f"Video information saved to: {info_path}")

    # 释放资源
    for clip in clips:
        clip.close()
    final_clip.close()

# 使用示例
if __name__ == "__main__":
    category = "Sports"  # 可以更改为其他大类
    n_per_label = 1  # 每个标签选择的视频数量
    video_dir = "archive/videos_val"
    output_dir = "composite_videos"
    label_file = "archive/kinetics400_val_list_videos.txt"
    csv_file = "archive/kinetics_400_labels.csv"

    create_composite_video(category, n_per_label, video_dir, output_dir, label_file, csv_file)