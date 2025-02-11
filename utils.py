from torchvision import models, transforms
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tsp_solver.greedy import solve_tsp
import os
import torch

def extract_frames(video_path : str) -> list:
    """
    Simple function to extract frames from a video file using OpenCV.
    Arguments:
        video_path (str): Path to the video file
    Returns:
        list: List of frames
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames



def extract_features(frames : list) -> np.ndarray:
    """
    Extract features from frames using pre-trained ResNet50 model trained on ImageNet 1K.
    Arguments:
        frames (list): List of frames
    Returns:
        np.ndarray: Features of frames
    """
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Define transform matching PyTorch's pre-trained ResNet50 preprocessing for ImageNet
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Preprocess each frame - apply the transform
    processed_frames = []
    for frame in frames:
        processed_frame = preprocess(frame)
        processed_frames.append(processed_frame)
    
    # Load pre-trained ResNet50 model
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights).to(device)
    model.eval()
    
    # Remove the classification layer (last layer). We only need the features which is the output
    # of the global average pooling layer (2nd to last layer)
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1])).to(device)
    
    # In our toy example, using a mini batch is not really necessary as we have a small number of frames
    # and not that high of a resolution, thus we can process all frames at once. However, in a real-world
    # scenario, using mini batches is preferable to avoid running out of memory
    features = []
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(processed_frames), batch_size):
            batch = torch.stack(processed_frames[i:i+batch_size])  # Shape before: (batch_size, 3, 224, 224)
            batch = batch.to(device)
            batch_features = feature_extractor(batch)  # Shape after: (batch_size, 2048, 1, 1)
            batch_features = batch_features.view(batch_features.size(0), -1)  # Flatten to shape: (batch_size, 2048)
            features.append(batch_features)
    
    features = torch.cat(features, dim=0)  # Concatenate all batch features

    return features.cpu().numpy()



def filter_relevant_frames(features : np.ndarray, eps : int  = 10, min_samples : int = 5) -> tuple:
    """
    Cluster features of frames using DBSCAN and filter relevant frames.
    Arguments:
        features (np.ndarray): Features of frames
        eps (int): DBSCAN hyperparameter. Maximum distance between two samples for one to be considered as in the neighborhood of the other
        min_samples (int): DBSCAN hyperparameter, The number of samples in a neighborhood for a point to be considered as a core point
    Returns:
        tuple: Indices of relevant and noise frames
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(features)
    labels = clustering.labels_

    # Identify the largest cluster (excluding outliers labeled -1)
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

    if len(unique_labels) == 0:
        # If no clusters found then either all frames are noise or there are no frames, in both cases return all frames
        print("No clusters found. Returning all frames as outliers.")
        return np.arange(len(features)), np.array([])
    
    largest_cluster = unique_labels[np.argmax(counts)]
    relevant_indices = np.where(labels == largest_cluster)[0]
    noise_indices = np.where(labels == -1)[0]
    
    return relevant_indices, noise_indices



def reorder_frames(features : np.ndarray, relevant_indices : np.ndarray) -> np.ndarray:
    """
    Reorder frames using TSP on cosine similarity matrix of features
    Arguments:
        features (np.ndarray): Features of frames
        relevant_indices (np.ndarray): Indices of relevant frames
    Returns:
        np.ndarray: Optimal order of frames
    """
    relevant_features = features[relevant_indices]

    # Compute cosine similarity matrix
    norms = np.linalg.norm(relevant_features, axis=1)
    similarity_matrix = np.dot(relevant_features, relevant_features.T) / np.outer(norms, norms)
    
    # Convert similarity to distance matrix (1 - similarity)
    distance_matrix = 1 - similarity_matrix
    
    # Solve TSP using a greedy heuristic
    path = solve_tsp(distance_matrix)
    
    return path



def reconstruct_video(input_video_path : str, output_video_path : str, denoised_frames_path : str, noise_frames_path : str) -> None:
    """
    Remove the noise frames from the input video and put the remaining frames in the optimal order. The feature
    extraction is done using a pre-trained ResNet50 model. The frames are clustered using DBSCAN, with
    the largest cluster considered as the relevant frames. The relevant frames are then reordered using TSP.
     
    Arguments:
        input_video_path (str): Path to the input video file
        output_video_path (str): Path to save the reconstructed video
        denoised_frames_path (str): Path to save the denoised and ordered frames
        noise_frames_path (str): Path to save the noise frames
    """
    # Extract frames from the input video
    frames = extract_frames(input_video_path)
    if not frames:
        raise ValueError("No frames extracted from the video.")
    
    # Extract features from frames using ResNet50
    features = extract_features(frames)
    
    # Filter relevant frames using DBSCAN
    relevant_indices, noise_indices = filter_relevant_frames(features)
    relevant_frames = [frames[i] for i in relevant_indices]
    bad_frames = [frames[i] for i in noise_indices]


    # Reorder the relevant frames (the ones that are not noise) using TSP
    optimal_order = reorder_frames(features, relevant_indices)
    ordered_frames = [relevant_frames[i] for i in optimal_order]
    
    # Save the reconstructed video, noise frames and ordered frames
    height, width, _ = ordered_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height))
    for frame in ordered_frames:
        out.write(frame)
    out.release()
    print("Reconstructed video saved to", output_video_path)

    # Ordered frames
    for i, frame in enumerate(ordered_frames):
        if not os.path.exists(denoised_frames_path):
            os.makedirs(denoised_frames_path)
        cv2.imwrite(f"{denoised_frames_path}/{i+1}_denoised_{optimal_order[i]}.jpg", frame)
    print("Ordered frames after denoising saved to", denoised_frames_path)

    # Noise frames
    for i, frame in enumerate(bad_frames):
        if not os.path.exists(noise_frames_path):
            os.makedirs(noise_frames_path)
        cv2.imwrite(f"{noise_frames_path}/noise_{i+1}.jpg", frame)
    print("Noise frames saved to", noise_frames_path)
    

