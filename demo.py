import os
import av
import torch
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import json
import gradio as gr
from pathlib import Path
import shutil
import logging
from PIL import Image
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load Kinetic-400 labels
kinetics_labels = pd.read_csv('kinetics_400_labels.csv', index_col='id')
KINETIC_400_LABELS = kinetics_labels['name'].tolist()

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
        patch_size = int(np.sqrt(attention.shape[0] - 1))
        att_map = attention[1:].reshape(patch_size, patch_size)
        att_resized = cv2.resize(att_map, (frame.shape[1], frame.shape[0]))
        
        att_smoothed = cv2.GaussianBlur(att_resized, (7, 7), 0)
        
        att_threshold = np.percentile(att_smoothed, 70)
        att_smoothed[att_smoothed < att_threshold] = 0
        
        att_norm = (att_smoothed - att_smoothed.min()) / (att_smoothed.max() - att_smoothed.min() + 1e-8)
        
        heatmap = cv2.applyColorMap(np.uint8(255 * att_norm), cv2.COLORMAP_JET)
        
        blend = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        
        return blend

def process_video(video_path, output_dir, extractor):
    logging.info(f"Processing video: {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    fps = video_stream.average_rate
    total_frames = video_stream.frames
    
    frames = []
    attention_buffer = []
    attention_data = []
    frame_count = 0
    all_logits = []
    temporal_smoothing_window = 3

    for frame in tqdm(container.decode(video=0), desc="Processing frames", total=total_frames):
        frame_rgb = frame.to_rgb().to_ndarray()
        frames.append(frame_rgb)
        
        if len(frames) == 8:
            spatial_attention, logits = extractor.extract_attention(frames)
            all_logits.append(logits)
            
            for i in range(8):
                attention = spatial_attention[0, i+1]
                attention_buffer.append(attention)
                
                if len(attention_buffer) >= temporal_smoothing_window:
                    smoothed_attention = np.mean(attention_buffer[-temporal_smoothing_window:], axis=0)
                    heatmap_frame = extractor.apply_attention_heatmap(frames[i], smoothed_attention)
                    
                    frame_filename = f"frame_{frame_count+1}_spatial_attention.png"
                    cv2.imwrite(os.path.join(frames_dir, frame_filename), cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2BGR))
                    
                    attention_data.append({
                        "frame_index": frame_count,
                        "max_attention": float(smoothed_attention[1:].max()),
                        "min_attention": float(smoothed_attention[1:].min()),
                        "mean_attention": float(smoothed_attention[1:].mean())
                    })
                    
                    frame_count += 1
            
            frames = frames[7:]

    # Process remaining frames
    if frames:
        padding = [frames[-1]] * (8 - len(frames))
        spatial_attention, logits = extractor.extract_attention(frames + padding)
        all_logits.append(logits)
        
        for i in range(len(frames)):
            attention = spatial_attention[0, i+1]
            attention_buffer.append(attention)
            
            smoothed_attention = np.mean(attention_buffer[-temporal_smoothing_window:], axis=0)
            heatmap_frame = extractor.apply_attention_heatmap(frames[i], smoothed_attention)
            
            frame_filename = f"frame_{frame_count+1}_spatial_attention.png"
            cv2.imwrite(os.path.join(frames_dir, frame_filename), cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2BGR))
            
            attention_data.append({
                "frame_index": frame_count,
                "max_attention": float(smoothed_attention[1:].max()),
                "min_attention": float(smoothed_attention[1:].min()),
                "mean_attention": float(smoothed_attention[1:].mean())
            })
            
            frame_count += 1

    # Save attention data
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(attention_data, f)

    # Calculate overall logits and predicted label
    overall_logits = np.mean(all_logits, axis=0)
    predicted_label = int(np.argmax(overall_logits))
    predicted_action = KINETIC_400_LABELS[predicted_label]

    logging.info(f"Video processing completed. Total frames: {frame_count}")
    return predicted_label, predicted_action, frames_dir, os.path.join(output_dir, "results.json")

def visualize_saliency(video_name, num_segments=8, results_dir=''):
    logging.info(f"Visualizing saliency for video: {video_name}")
    # Load data
    with open(os.path.join(results_dir, "results.json"), 'r') as f:
        attention_data = json.load(f)
    
    # Extract temporal attention
    temporal_attention = np.array([frame['mean_attention'] for frame in attention_data])
    
    # Normalize temporal attention
    temporal_attention = (temporal_attention - temporal_attention.min()) / (temporal_attention.max() - temporal_attention.min())
    
    # Select frames to visualize
    total_frames = len(attention_data)
    frame_indices = np.linspace(0, total_frames-1, num_segments, dtype=int)
    
    # Create figure with adjusted dimensions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [1, 0.7]})
    fig.suptitle(f"Temporal Saliency and Key Frames", fontsize=12)
    
    # Plot temporal saliency
    ax1.plot(range(total_frames), temporal_attention, color='blue', alpha=0.5)
    ax1.scatter(frame_indices, temporal_attention[frame_indices], color='red', s=30, zorder=5)
    for idx in frame_indices:
        ax1.axvline(x=idx, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel("Frame Number", fontsize=10)
    ax1.set_ylabel("Temporal Saliency", fontsize=10)
    ax1.set_xlim(0, total_frames-1)
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='both', which='major', labelsize=8)
    
    # Display key frames
    for i, frame_idx in enumerate(frame_indices):
        # Read the frame
        frame_path = os.path.join(results_dir, 'frames', f"frame_{frame_idx+1}_spatial_attention.png")
        frame = cv2.imread(frame_path)
        if frame is None:
            logging.warning(f"Frame not found: {frame_path}")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add frame to the plot
        ax_sub = ax2.inset_axes([i/num_segments, 0.1, 1/num_segments - 0.01, 0.8], transform=ax2.transAxes)
        ax_sub.imshow(frame)
        ax_sub.axis('off')
        ax_sub.set_title(f"Frame {frame_idx+1}", fontsize=8)
    
    ax2.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)  # Adjust space between subplots
    output_path = os.path.join(results_dir, f"{video_name}_temporal_saliency_visualization.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Saliency visualization saved to: {output_path}")
    return output_path

def create_heatmap_gif(output_dir, video_name, max_frames=50, scale_factor=0.5):
    logging.info(f"Creating heatmap GIF for: {video_name}")
    frames_dir = os.path.join(output_dir, 'frames')
    output_gif_path = os.path.join(output_dir, f"{video_name}_heatmap.gif")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('_spatial_attention.png')])
    if not frame_files:
        raise ValueError("No frames found to create GIF")
    
    # Reduce number of frames if exceeds max_frames
    if len(frame_files) > max_frames:
        frame_files = frame_files[::len(frame_files)//max_frames][:max_frames]
    
    images = []
    for frame_file in tqdm(frame_files, desc="Creating GIF"):
        with Image.open(os.path.join(frames_dir, frame_file)) as img:
            # Resize the image
            width, height = img.size
            new_size = (int(width * scale_factor), int(height * scale_factor))
            img_resized = img.resize(new_size, Image.LANCZOS)
            images.append(img_resized)
    
    logging.info(f"Saving GIF with {len(images)} frames")
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
    
    logging.info(f"Heatmap GIF created: {output_gif_path}")
    return output_gif_path

def create_heatmap_video(output_dir, video_name):
    logging.info(f"Creating heatmap video for: {video_name}")
    frames_dir = os.path.join(output_dir, 'frames')
    output_video_path = os.path.join(output_dir, f"{video_name}_heatmap.mp4")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('_spatial_attention.png')])
    if not frame_files:
        raise ValueError("No frames found to create video")
    
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
    
    for frame_file in tqdm(frame_files, desc="Creating video"):
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        out.write(frame)
    
    out.release()
    
    logging.info(f"Heatmap video created: {output_video_path}")
    return output_video_path

def process_video_gradio(video_file, progress=gr.Progress()):
    logging.info("Starting video processing")
    # Create a temporary directory to save the uploaded video and processing results
    temp_dir = Path("temp_output")
    temp_dir.mkdir(exist_ok=True)
    
    # Save the uploaded video
    video_path = temp_dir / "input_video.mp4"
    
    # Check if video_file is a string (file path) or a UploadFile object
    if isinstance(video_file, str):
        shutil.copy(video_file, video_path)
    else:
        # Assuming video_file is a UploadFile object
        with open(video_path, "wb") as f:
            f.write(video_file.read())
    
    logging.info(f"Video saved to: {video_path}")
    
    progress(0, desc="Initializing")
    
    # Initialize AttentionExtractor
    extractor = AttentionExtractor('facebook/timesformer-base-finetuned-k400')
    
    progress(0.1, desc="Processing video")
    # Process video
    output_dir = temp_dir / "output"
    predicted_label, predicted_action, frames_dir, json_path = process_video(str(video_path), str(output_dir), extractor)
    
    progress(0.7, desc="Generating visualization")
    # Generate visualization
    video_name = video_path.stem
    visualization_path = visualize_saliency(video_name, results_dir=str(output_dir))
    
    progress(0.8, desc="Creating heatmap GIF")
    # Generate GIF with heatmap
    heatmap_gif_path = create_heatmap_gif(str(output_dir), video_name)
    
    progress(0.9, desc="Creating heatmap video")
    # Generate MP4 with heatmap
    heatmap_video_path = create_heatmap_video(str(output_dir), video_name)
    
    progress(1.0, desc="Completed")
    logging.info("Video processing completed")
    return predicted_label, predicted_action, heatmap_gif_path, json_path, visualization_path, heatmap_video_path

iface = gr.Interface(
    fn=process_video_gradio,
    inputs=gr.File(label="Upload MP4 Video"),
    outputs=[
        gr.Number(label="Predicted Label ID"),
        gr.Text(label="Predicted Action"),
        gr.Image(label="Heatmap Animation", image_mode='gif'),
        gr.File(label="JSON Results"),
        gr.Image(label="Temporal Saliency Visualization"),
        gr.File(label="Download Heatmap Video (MP4)")
    ],
    title="Video Attention Analysis",
    description="Upload an MP4 video to analyze attention and generate visualizations."
)

if __name__ == "__main__":
    iface.launch()