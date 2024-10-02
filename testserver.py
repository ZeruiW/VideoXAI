import requests

# 上传视频文件并处理
url = "http://localhost:6313/process_video/"
files = {"file": open("archive/videos_val/__vzEs2wzdQ.mp4", "rb")}
response = requests.post(url, files=files)
print(response.json())

# 下载结果
video_name = "your_video_name"
requests.get(f"http://localhost:6313/download/heatmap_video/{video_name}")
requests.get(f"http://localhost:6313/download/json_data/{video_name}")
requests.get(f"http://localhost:6313/download/visualization/{video_name}")
