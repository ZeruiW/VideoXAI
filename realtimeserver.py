import os
import torch
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor
import cv2
import json
import logging
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import zmq
import av
from threading import Thread
import time
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttentionExtractor:
    def __init__(self, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = TimesformerForVideoClassification.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.id2label = self.model.config.id2label

    def extract_attention(self, frames):
        inputs = self.image_processor(frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        last_layer_attention = outputs.attentions[-1]
        spatial_attention = last_layer_attention.mean(1)
        return spatial_attention.cpu().numpy(), outputs.logits.cpu().numpy()

    def apply_attention_heatmap(self, frame, attention, predicted_label):
        att_map = attention[1:].reshape(int(np.sqrt(attention.shape[0]-1)), -1)
        att_resized = cv2.resize(att_map, (frame.shape[1], frame.shape[0]))
        att_norm = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * att_norm), cv2.COLORMAP_JET)
        blend = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        
        label_text = f"Predicted: {self.id2label[predicted_label]}"
        cv2.putText(blend, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return blend

class VideoServer:
    def __init__(self, model_name, output_dir, port=6000):
        self.extractor = AttentionExtractor(model_name)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.frames_dir = os.path.join(self.output_dir, 'frames')
        os.makedirs(self.frames_dir, exist_ok=True)
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.bind(f"tcp://*:{port}")
        
        self.running = True
        self.frame_buffer = []
        self.attention_buffer = []
        self.all_logits = []
        self.attention_data = []
        self.frame_count = 0
        self.predicted_labels = []

        self.output_video = None
        self.output_stream = None

    def receive_frames(self):
        while self.running:
            message = self.socket.recv_pyobj()
            if message == "STOP":
                self.running = False
                break
            frames, fps = message
            if self.output_video is None:
                self.initialize_output_video(frames[0].shape[:2][::-1], fps)
            self.process_frames(frames)

    def initialize_output_video(self, resolution, fps):
        self.output_video = av.open(os.path.join(self.output_dir, 'output.mp4'), mode='w')
        self.output_stream = self.output_video.add_stream('h264', rate=fps)
        self.output_stream.width, self.output_stream.height = resolution
        self.output_stream.pix_fmt = 'yuv420p'

    def process_frames(self, frames):
        self.frame_buffer.extend(frames)
        
        while len(self.frame_buffer) >= 8:
            batch_frames = self.frame_buffer[:8]
            spatial_attention, logits = self.extractor.extract_attention(batch_frames)
            self.all_logits.append(logits)
            predicted_label = int(np.argmax(logits))
            self.predicted_labels.append(predicted_label)
            
            attention = spatial_attention[0, -1]
            self.attention_buffer.append(attention)
            
            smoothed_attention = np.mean(self.attention_buffer[-5:], axis=0) if len(self.attention_buffer) >= 5 else attention
            
            for i, frame in enumerate(batch_frames):
                heatmap_frame = self.extractor.apply_attention_heatmap(frame, smoothed_attention, predicted_label)
                
                if i == 7:  # Save the last frame of each batch
                    frame_filename = f"frame_{self.frame_count + i}_spatial_attention.png"
                    cv2.imwrite(os.path.join(self.frames_dir, frame_filename), cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2BGR))
                
                # Write frame to output video
                out_frame = av.VideoFrame.from_ndarray(heatmap_frame, format='rgb24')
                packet = self.output_stream.encode(out_frame)
                self.output_video.mux(packet)
            
            self.attention_data.append({
                "frame_index": self.frame_count,
                "max_attention": float(smoothed_attention[1:].max()),
                "min_attention": float(smoothed_attention[1:].min()),
                "mean_attention": float(smoothed_attention[1:].mean()),
                "predicted_label": self.extractor.id2label[predicted_label]
            })
            
            self.frame_count += 8
            self.frame_buffer = self.frame_buffer[8:]

    def create_temporal_visualization(self):
        temporal_attention = np.array([frame['mean_attention'] for frame in self.attention_data])
        frame_indices = np.array([frame['frame_index'] for frame in self.attention_data])
        
        plt.figure(figsize=(20, 10))
        sns.set_style("whitegrid")
        
        plt.plot(frame_indices, temporal_attention, color='blue', linewidth=2, label='Attention')
        
        # Apply smoothing only if we have enough data points
        if len(temporal_attention) > 5:
            try:
                window_length = min(len(temporal_attention) - 1, 51)  # Must be odd and less than data length
                window_length = window_length - 1 if window_length % 2 == 0 else window_length
                polyorder = min(3, window_length - 1)  # Ensure polyorder is less than window_length
                temporal_attention_smoothed = savgol_filter(temporal_attention, window_length, polyorder)
                plt.plot(frame_indices, temporal_attention_smoothed, color='red', linewidth=2, label='Smoothed Attention')
            except Exception as e:
                print(f"Warning: Could not apply smoothing filter. Error: {e}")
                print("Proceeding with unsmoothed data.")
        
        plt.xlabel("Frame Number", fontsize=14)
        plt.ylabel("Temporal Attention", fontsize=14)
        plt.title("Temporal Attention Over Time", fontsize=16)
        plt.legend(fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "temporal_visualization.png"), dpi=300)
        plt.close()

        print(f"Temporal visualization saved to: {os.path.join(self.output_dir, 'temporal_visualization.png')}")

    def create_frames_visualization(self):
        num_frames = len(self.attention_data)
        cols = 4
        rows = (num_frames + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
        fig.suptitle("Sampled Frames with Attention Heatmaps", fontsize=16)
        
        for i, frame_data in enumerate(self.attention_data):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            frame_path = os.path.join(self.frames_dir, f"frame_{frame_data['frame_index']}_spatial_attention.png")
            if os.path.exists(frame_path):
                frame = plt.imread(frame_path)
                ax.imshow(frame)
                ax.axis('off')
                ax.set_title(f"Frame: {frame_data['frame_index']}\nLabel: {frame_data['predicted_label']}\nAttention: {frame_data['mean_attention']:.4f}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "Frame not found", ha='center', va='center')
                ax.axis('off')
        
        # Remove empty subplots
        for i in range(num_frames, rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axes[row, col] if rows > 1 else axes[col])
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "frames_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        Thread(target=self.receive_frames).start()

        while self.running:
            time.sleep(1)

        print("Processing completed. Creating visualizations...")
        self.create_temporal_visualization()
        self.create_frames_visualization()
        
        # Save attention data
        with open(os.path.join(self.output_dir, "attention_data.json"), 'w') as f:
            json.dump(self.attention_data, f)
        
        # Finalize output video
        packet = self.output_stream.encode(None)
        self.output_video.mux(packet)
        self.output_video.close()
        
        overall_logits = np.mean(self.all_logits, axis=0)
        final_predicted_label = int(np.argmax(overall_logits))
        print(f"Final Predicted Label: {self.extractor.id2label[final_predicted_label]}")
        print(f"Temporal visualization saved to: {os.path.join(self.output_dir, 'temporal_visualization.png')}")
        print(f"Frames visualization saved to: {os.path.join(self.output_dir, 'frames_visualization.png')}")
        print(f"Output video saved to: {os.path.join(self.output_dir, 'output.mp4')}")

if __name__ == "__main__":
    server = VideoServer('facebook/timesformer-base-finetuned-k400', 'Realtime')
    print("Server started on port 6000")
    server.run()