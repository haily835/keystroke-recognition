import cv2
import pandas as pd
import os
import numpy as np

def extract_and_merge_frames(video_path, csv_path, frames_before=3, frames_after=4):
    # Read the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error: Could not open video file {video_path}")
    
    # Get video properties
    fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"FPS: {fps}, Total frames: {total_frames}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create output directory if it doesn't exist
    os.makedirs("output_videos", exist_ok=True)

    for index, row in df.iterrows():
        key = row['Key']
        frame_number = row['Frame'] 

        frames = []
        for i in range(-frames_before, frames_after + 1):
            frame_index = frame_number + i
            if 0 <= frame_index < total_frames:
                # Set frame position
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)

        if frames:
            # Stack frames horizontally
            merged_frame = np.stack(frames)
            
            output_filename = f"{frame_number}_{key}.mp4"
            output_path = os.path.join("output_videos", output_filename)
            
            # Define video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            height, width = frames[0].shape[:2]
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Write frames
            for frame in merged_frame:
                out.write(frame)
            
            out.release()
            print(f"Saved: {output_filename}")
    
    # Release video capture
    cap.release()

# Example usage
extract_and_merge_frames('video-0.mp4', 'video-0.csv')