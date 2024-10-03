import cv2
import numpy as np
import zmq
import time
from threading import Thread
import av
import argparse

class VideoClient:
    def __init__(self, server_address, video_path):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(server_address)
        self.running = True
        self.video_path = video_path

    def send_video(self):
        container = av.open(self.video_path)
        stream = container.streams.video[0]
        fps = stream.average_rate
        
        frames = []
        frame_count = 0
        for frame in container.decode(video=0):
            if not self.running:
                break
            frame_rgb = frame.to_rgb().to_ndarray()
            frames.append(frame_rgb)
            frame_count += 1
            
            if frame_count % int(fps / 2) == 0:  # 每0.5秒发送一次
                self.socket.send_pyobj((frames, fps))
                frames = []
        
        if frames:  # 发送剩余的帧
            self.socket.send_pyobj((frames, fps))
        
        container.close()
        self.socket.send_pyobj("STOP")
        print("Video processing completed. Sent STOP signal to server.")

    def run(self):
        Thread(target=self.send_video).start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping client...")
            self.running = False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Client")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--server", type=str, default="localhost", help="Server address")
    args = parser.parse_args()

    server_address = f"tcp://{args.server}:6000"
    client = VideoClient(server_address, args.video)
    client.run()