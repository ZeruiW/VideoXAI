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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AttentionExtractor:
    def __init__(self, model_name, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = TimesformerForVideoClassification.from_pretrained(model_name)
        self.model.to(device)
        self.device = device
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)

    def extract_attention(self, frames):
        inputs = self.image_processor(frames, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        last_layer_attention = outputs.attentions[-1]
        spatial_attention = last_layer_attention.mean(1)
        return spatial_attention.cpu().numpy(), outputs.logits.cpu().numpy()

    def apply_attention_heatmap(self, frame, attention):
        att_map = attention[1:].reshape(int(np.sqrt(attention.shape[0]-1)), -1)
        att_resized = cv2.resize(att_map, (frame.shape[1], frame.shape[0]))
        att_norm = (att_resized - att_resized.min()) / (att_resized.max() - att_resized.min())
        heatmap = cv2.applyColorMap(np.uint8(255 * att_norm), cv2.COLORMAP_JET)
        blend = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        return blend

class VideoServer:
    def __init__(self, model_name, output_dir, port=6000, sampling_rate=15):
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
        self.sampling_rate = sampling_rate
        self.predicted_labels = []

    def receive_frames(self):
        while self.running:
            message = self.socket.recv_pyobj()
            if message == "STOP":
                self.running = False
                break
            frames = [cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR) for frame_bytes in message]
            self.process_frames(frames)

    def process_frames(self, frames):
        self.frame_buffer.extend(frames)
        
        while len(self.frame_buffer) >= 8:
            if self.frame_count % self.sampling_rate == 0:
                batch_frames = self.frame_buffer[:8]
                spatial_attention, logits = self.extractor.extract_attention(batch_frames)
                self.all_logits.append(logits)
                predicted_label = int(np.argmax(logits))
                self.predicted_labels.append(predicted_label)
                
                attention = spatial_attention[0, -1]
                self.attention_buffer.append(attention)
                
                if len(self.attention_buffer) >= 5:
                    smoothed_attention = np.mean(self.attention_buffer[-5:], axis=0)
                    heatmap_frame = self.extractor.apply_attention_heatmap(batch_frames[-1], smoothed_attention)
                    
                    frame_filename = f"frame_{self.frame_count}_spatial_attention.png"
                    cv2.imwrite(os.path.join(self.frames_dir, frame_filename), cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2BGR))
                    
                    self.attention_data.append({
                        "frame_index": self.frame_count,
                        "max_attention": float(smoothed_attention[1:].max()),
                        "min_attention": float(smoothed_attention[1:].min()),
                        "mean_attention": float(smoothed_attention[1:].mean()),
                        "predicted_label": predicted_label
                    })
            
            self.frame_count += 1
            self.frame_buffer = self.frame_buffer[1:]

    def create_summary_visualization(self):
        temporal_attention = np.array([frame['mean_attention'] for frame in self.attention_data])
        frame_indices = np.array([frame['frame_index'] for frame in self.attention_data])
        
        window_length = min(len(temporal_attention) // 2 * 2 + 1, 21)
        temporal_attention_smoothed = savgol_filter(temporal_attention, window_length, 3)
        
        temporal_attention_smoothed = (temporal_attention_smoothed - temporal_attention_smoothed.min()) / (temporal_attention_smoothed.max() - temporal_attention_smoothed.min())
        
        peaks, _ = find_peaks(temporal_attention_smoothed, distance=len(temporal_attention_smoothed)//8)
        if len(peaks) < 8:
            additional_frames = np.linspace(0, len(temporal_attention_smoothed)-1, 8-len(peaks), dtype=int)
            key_frame_indices = np.sort(np.concatenate([peaks, additional_frames]))
        else:
            key_frame_indices = peaks[:8]
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 2])
        
        # Plot temporal saliency
        ax1 = plt.subplot(gs[0])
        ax1.plot(frame_indices, temporal_attention_smoothed, color='blue', alpha=0.7, linewidth=2)
        ax1.scatter(frame_indices[key_frame_indices], temporal_attention_smoothed[key_frame_indices], color='red', s=100, zorder=5)
        for idx in key_frame_indices:
            ax1.axvline(x=frame_indices[idx], color='gray', linestyle='--', alpha=0.5)
        ax1.set_xlabel("Frame Number", fontsize=12)
        ax1.set_ylabel("Temporal Saliency", fontsize=12)
        ax1.set_xlim(frame_indices[0], frame_indices[-1])
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='both', which='major', labelsize=10)
        ax1.set_title("Temporal Saliency", fontsize=14)
        
        # Plot predicted labels
        ax2 = plt.subplot(gs[1])
        ax2.plot(frame_indices, self.predicted_labels[:len(frame_indices)], color='green', alpha=0.7, linewidth=2)
        ax2.set_xlabel("Frame Number", fontsize=12)
        ax2.set_ylabel("Predicted Label", fontsize=12)
        ax2.set_xlim(frame_indices[0], frame_indices[-1])
        ax2.tick_params(axis='both', which='major', labelsize=10)
        ax2.set_title("Predicted Labels Over Time", fontsize=14)
        
        # Display key frames
        ax3 = plt.subplot(gs[2])
        ax3.axis('off')
        for i, idx in enumerate(key_frame_indices):
            frame_path = os.path.join(self.frames_dir, f"frame_{self.attention_data[idx]['frame_index']}_spatial_attention.png")
            if os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ax_sub = ax3.inset_axes([i/8, 0, 1/8 - 0.01, 1], transform=ax3.transAxes)
                ax_sub.imshow(frame)
                ax_sub.axis('off')
                ax_sub.set_title(f"Frame {self.attention_data[idx]['frame_index']}", fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "summary_visualization.png"), dpi=300, bbox_inches='tight')
        plt.close()

    def run(self):
        Thread(target=self.receive_frames).start()

        while self.running:
            time.sleep(1)

        print("Processing completed. Creating summary...")
        self.create_summary_visualization()
        
        # Save attention data
        with open(os.path.join(self.output_dir, "attention_data.json"), 'w') as f:
            json.dump(self.attention_data, f)
        
        overall_logits = np.mean(self.all_logits, axis=0)
        final_predicted_label = int(np.argmax(overall_logits))
        print(f"Final Predicted Label: {final_predicted_label}")
        print(f"Summary visualization saved to: {os.path.join(self.output_dir, 'summary_visualization.png')}")

if __name__ == "__main__":
    server = VideoServer('facebook/timesformer-base-finetuned-k400', 'Realtime')
    print("Server started on port 6000")
    server.run()