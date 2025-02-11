# Original Video Reconstruction from Corrupted Footage

This project aims to reconstruct an original video from a corrupted version where frames have been shuffled and mixed with irrelevant images. The proposed algorithm:
- Extracts frames from the corrupted video.
- Extracts features from frames using a pre-trained ResNet50 model.
- Identifies and filters out irrelevant frames using DBSCAN clustering.
- Reorders the remaining frames using the Traveling Salesman Problem (TSP) approach.
- Reconstructs the video using the filtered and reordered frames.

The method is designed to be generalizable to other similarly corrupted videos.

## Setup

### 1. Create a conda environment
Using conda:

```bash
conda create -n video_reconstruction python=3.12.7
conda activate video_reconstruction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage & Outputs

Default arguments have been set for the provided `corrupted_video.mp4` file. To run the algorithm on this video, simply run:

```bash
python videoReconstruction.py
```

The script will then generate the following outputs:
- Reconstructed Video: Stored in `Data/reconstructed_video.mp4`.
- Denoised Ordered Frames: Saved in the `Data/Denoised_ordered_frames/` directory.
- Noise Frames: Stored separately in `Data/Noise_frames/`.


## Files
- `requirements.txt`
- `utils.py` - Contains functions for frame extraction, feature extraction, DBSCAN clustering, TSP-based frame reordering and video reconstruction.
- `videoReconstruction.py` - The main script that processes the corrupted video and reconstructs it.
- `Data/` - Directory containing input and output videos, along with intermediate frame processing results.

## Pipeline

ResNet50 is chosen for feature extraction because it maps images into a latent feature space where visually similar images have closer representations. This allows meaningful comparisons between frames, enabling the identification of relevant frames while filtering out noise.

DBSCAN is then applied to the extracted feature vectors. Its density-based approach helps in identifying the dominant cluster of frames that belong to the original sequence while isolating outliers that are unrelated noise frames. For different videos, the hyperparameters (epsilon and min_samples) may need to be adjusted based on how distinguishable the noise images are. For less distinguishable noise frames, lowering epsilon or increasing min_samples can refine the separation between the main sequence and outliers, leading to more accurate noise removal.

Finally, the TSP solver is used to reorder the relevant frames. Since similar frames are expected to be temporally adjacent in a coherent video, the shortest path in the feature space should closely approximate the correct chronological order. By minimizing the overall distance between consecutive frames, we reconstruct a video sequence that closely matches the original structure.

# Author
Efstathios Chatziloizos


