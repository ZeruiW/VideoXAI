import os
import av
import torch
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import json
import logging
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import shutil
import uvicorn
from pathlib import Path

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

def process_video(video_path, output_dir, extractor):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    container = av.open(video_path)
    video_stream = container.streams.video[0]
    fps = video_stream.average_rate
    total_frames = video_stream.frames
    
    frames = []
    attention_data = []
    frame_count = 0
    all_logits = []

    for frame in tqdm(container.decode(video=0), desc="Processing frames", total=total_frames):
        frame_rgb = frame.to_rgb().to_ndarray()
        frames.append(frame_rgb)
        
        if len(frames) == 8:
            spatial_attention, logits = extractor.extract_attention(frames)
            all_logits.append(logits)
            
            for i in range(8):
                attention = spatial_attention[0, i+1]
                heatmap_frame = extractor.apply_attention_heatmap(frames[i], attention)
                
                frame_filename = f"frame_{frame_count+1}_spatial_attention.png"
                cv2.imwrite(os.path.join(frames_dir, frame_filename), cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2BGR))
                
                attention_data.append({
                    "frame_index": frame_count,
                    "max_attention": float(attention[1:].max()),
                    "min_attention": float(attention[1:].min()),
                    "mean_attention": float(attention[1:].mean())
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
            heatmap_frame = extractor.apply_attention_heatmap(frames[i], attention)
            
            frame_filename = f"frame_{frame_count+1}_spatial_attention.png"
            cv2.imwrite(os.path.join(frames_dir, frame_filename), cv2.cvtColor(heatmap_frame, cv2.COLOR_RGB2BGR))
            
            attention_data.append({
                "frame_index": frame_count,
                "max_attention": float(attention[1:].max()),
                "min_attention": float(attention[1:].min()),
                "mean_attention": float(attention[1:].mean())
            })
            
            frame_count += 1

    # Save attention data
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(attention_data, f)

    # Calculate overall logits and predicted label
    overall_logits = np.mean(all_logits, axis=0)
    predicted_label = int(np.argmax(overall_logits))

    return predicted_label

def visualize_saliency(video_name, num_segments=8, results_dir=''):
    try:
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
                print(f"Frame not found: {frame_path}")
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
        
        print(f"Temporal saliency visualization saved to: {output_path}")
    except Exception as e:
        print(f"Error in visualize_saliency for video {video_name}: {str(e)}")
        raise

def create_heatmap_video(output_dir, video_name):
    frames_dir = os.path.join(output_dir, 'frames')
    output_video_path = os.path.join(output_dir, f"{video_name}_heatmap.mp4")
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('_spatial_attention.png')])
    if not frame_files:
        raise ValueError("No frames found to create video")
    
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))
    
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(frames_dir, frame_file))
        out.write(frame)
    
    out.release()
    return output_video_path

app = FastAPI()

@app.post("/process_video/")
async def process_video_endpoint(file: UploadFile = File(...)):
    # Create a temporary directory to save the uploaded video and processing results
    temp_dir = Path("temp_output")
    temp_dir.mkdir(exist_ok=True)
    
    # Save the uploaded video
    video_path = temp_dir / file.filename
    with video_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Initialize AttentionExtractor
    extractor = AttentionExtractor('facebook/timesformer-base-finetuned-k400')
    
    # Process video
    output_dir = temp_dir / "output"
    predicted_label = process_video(str(video_path), str(output_dir), extractor)
    
    # Generate visualization
    video_name = video_path.stem
    visualize_saliency(video_name, results_dir=str(output_dir))
    
    # Generate video with heatmap
    heatmap_video_path = create_heatmap_video(str(output_dir), video_name)
    
    # Prepare the result
    result = {
        "label": predicted_label,
        "heatmap_video": FileResponse(heatmap_video_path, media_type="video/mp4", filename=f"{video_name}_heatmap.mp4"),
        "json_data": FileResponse(output_dir / "results.json", media_type="application/json", filename="results.json"),
        "visualization": FileResponse(output_dir / f"{video_name}_temporal_saliency_visualization.png", media_type="image/png", filename=f"{video_name}_visualization.png")
    }
    
    return JSONResponse(content={
        "label": predicted_label,
        "message": "Processing complete. Use the following endpoints to download the results:",
        "heatmap_video_url": f"/download/heatmap_video/{video_name}",
        "json_data_url": f"/download/json_data/{video_name}",
        "visualization_url": f"/download/visualization/{video_name}"
    })

@app.get("/download/heatmap_video/{video_name}")
async def download_heatmap_video(video_name: str):
    return FileResponse(f"temp_output/output/{video_name}_heatmap.mp4", media_type="video/mp4", filename=f"{video_name}_heatmap.mp4")

@app.get("/download/json_data/{video_name}")
async def download_json_data(video_name: str):
    return FileResponse(f"temp_output/output/results.json", media_type="application/json", filename="results.json")

@app.get("/download/visualization/{video_name}")
async def download_visualization(video_name: str):
    return FileResponse(f"temp_output/output/{video_name}_temporal_saliency_visualization.png", media_type="image/png", filename=f"{video_name}_visualization.png")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6313)