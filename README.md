
## The supplementary experiments results
## Results Demonstration
## https://drive.google.com/file/d/1R8r2ZDcuikLu9s4CeeYZgW4bDxR_r5LD/view?usp=sharing

## Overview

This repository provides an implementation for generating SHAP values for temporal segments in video classification tasks using a Timesformer model. It includes tools for processing videos, computing Shapley values, and evaluating performance metrics.

## Table of Contents


  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Configuration](#configuration)
    - [Initialization](#initialization)
    - [Video Processing](#video-processing)
    - [SHAP Calculation](#shap-calculation)
    - [Performance Evaluation](#performance-evaluation)
  - [Results](#results)
  - [Contributing](#contributing)

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- av
- numpy
- scikit-learn
- psutil
- tqdm

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/temporal-shap.git
    cd temporal-shap
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Configuration

Modify the `config` dictionary in the script to set your parameters:

```python
config = {
    "model_name": "facebook/timesformer-base-finetuned-k400",
    "image_processor_name": "MCG-NJU/videomae-base-finetuned-kinetics",
    "num_samples": 100,
    "num_classes": 400,
    "num_samples_per_class": 25,
    "video_list_path": "archive/kinetics400_val_list_videos.txt",
    "video_directory": "archive/zoom_blur",
    "use_exact": True
}
```

### Initialization

Initialize the video processor and SHAP calculator:

```python
video_processor = VideoProcessor(config["model_name"], config["image_processor_name"])
shap_calculator = TemporalShap(num_samples=config["num_samples"])
```

### Video Processing

Process the videos and compute predictions and SHAP values:

```python
sampled_files = [...]  # List of video filenames
true_labels = [...]  # Corresponding true labels

video_data = process_videos(video_processor, shap_calculator, sampled_files, true_labels, use_exact=config["use_exact"])
```

### SHAP Calculation

Calculate SHAP values for the segments:

```python
sv_true_label = shap_calculator.approximate_shapley_values(segment_outputs, true_label)
sv_video_pred = shap_calculator.approximate_shapley_values(segment_outputs, video_pred_label)
```

### Performance Evaluation

Compute performance metrics:

```python
accuracy, precision, recall, f1 = compute_metrics(video_data)
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

save_performance_metrics(accuracy, precision, recall, f1, time_consumed, cpu_energy_consumed, gpu_energy_consumed, filename="performance.json")
```

## Results

Results are saved in `results.json` and performance metrics are saved in `performance.json`. You can load and inspect them for detailed analysis.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your changes.

