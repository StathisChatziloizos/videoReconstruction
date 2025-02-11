import utils
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconstruct a video using denoised and reordered frames.")
    parser.add_argument('--data_path', type=str, default="Data", help='Path to the data directory')
    parser.add_argument('--input_file', type=str, default="corrupted_video.mp4", help='Name of the input video file')
    parser.add_argument('--output_file', type=str, default="reconstructed_video.mp4", help='Name of the output video file')
    parser.add_argument('--denoised_frames_dir', type=str, default="Denoised_ordered_frames", help='Directory of denoised and ordered frames')
    parser.add_argument('--noise_frames_dir', type=str, default="Noise_frames", help='Directory of noise frames')

    args = parser.parse_args()

    input_path = f"{args.data_path}/{args.input_file}"
    output_path = f"{args.data_path}/{args.output_file}"
    denoised_frames_path = f"{args.data_path}/{args.denoised_frames_dir}"
    noise_frames_path = f"{args.data_path}/{args.noise_frames_dir}"

    # Reconstruct the video using the reordered frames, having removed the noise frames
    utils.reconstruct_video(input_path, output_path, denoised_frames_path, noise_frames_path)
