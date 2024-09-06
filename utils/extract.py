import cv2
import os
from datetime import datetime
import torchvision
import torch

def convert_timestamp_to_local_time(timestamp_seconds):
    # Convert timestamp (seconds from 1970) to a local datetime object
    local_datetime = datetime.fromtimestamp(timestamp_seconds)
    
    # Format it to a human-readable string (local time)
    formatted_local_datetime = local_datetime.strftime('%Y-%m-%d %H:%M:%S %Z')
    
    return formatted_local_datetime


def get_frame_at_timestamp(video_path, timestamp_ms):
    # Get the video's end time (modified time in seconds)
    video_end_time = os.path.getmtime(video_path)
    
    # Convert the timestamp (ms since 1970) to seconds
    timestamp_sec = timestamp_ms / 1000.0
    print('timestamp_sec: ', convert_timestamp_to_local_time(timestamp_sec))
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
    duration_sec = total_frames / fps  # Duration of the video in seconds
    
    # Calculate the video start time based on the modified time and duration
    video_start_time = video_end_time - duration_sec
    print('video_start_time: ', convert_timestamp_to_local_time(video_start_time))
    print('video_end_time: ', convert_timestamp_to_local_time(video_end_time))
    print('duration_sec: ', duration_sec)
    print('fps: ', fps)
    
    # Check if the timestamp is before the video started or after it ended
    if timestamp_sec < video_start_time or timestamp_sec > video_end_time:
        print("Error: Timestamp is outside the video duration.")
        return None
    
    # Calculate the relative time (seconds from the start of the video)
    relative_time = timestamp_sec - video_start_time
    
    # Calculate the corresponding frame number
    frame_number = int(fps * relative_time)
    
    return frame_number, fps, cap


def get_surrounding_frames(video_path, timestamp_ms, frames_before=3, frames_after=4):
    # Get the frame at the given timestamp and video properties
    frame_number, fps, cap = get_frame_at_timestamp(video_path, timestamp_ms)

    if frame_number is None:
        return
    
    # Define the range of frames to retrieve (3 frames before and 4 frames after)
    start_frame = max(0, frame_number - frames_before)
    end_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), frame_number + frames_after)
    
    # Store the frames
    frames = []
    
    # Loop through the desired range and collect frames
    for f_num in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_num)
        ret, frame = cap.read()
        if ret:
            # Convert the frame to RGB format because OpenCV reads frames as BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        else:
            print(f"Error reading frame {f_num}")
    
    # Release the video capture object
    cap.release()
    
    return frames, fps


def save_short_video_with_torchvision(frames, output_path, target_fps=2):
    # Convert frames to a tensor and prepare it for torchvision's write_video
    frames_tensor = torch.tensor(frames) # From (N, H, W, C) to (N, C, H, W)
    
    # Save the video using torchvision
    torchvision.io.write_video(output_path, frames_tensor, target_fps)
    print(f"Video saved to {output_path}")


# Example usage
video_path = "/Users/haily/Desktop/VID_20240905_224213.mp4"
timestamp_ms = 1725590413488  # Replace with your timestamp in milliseconds (from 1970)
frames, fps = get_surrounding_frames(video_path, timestamp_ms)

if frames:
    output_path = "short_video_torchvision.mp4"
    save_short_video_with_torchvision(frames, output_path, target_fps=2)
