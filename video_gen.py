import cv2
import pandas as pd
import os
import torch
import torchvision.io as tvio
from utils.realtime_util import remove_consecutive_letters
from tqdm import tqdm
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

idx = 6
# File paths and threshold settings
frame_folder = f'datasets/KeyVR/raw_frames/video_{idx}'  # Folder containing frames (e.g., 'frame_0.jpg')
first_csv_path = f'datasets/KeyVR/labels/video_{idx}.csv'
second_csv_path = f'ckpts/HyperGT/stream_results/{idx}.csv'
lm_path = f'datasets/KeyVR/landmarks/video_{idx}.pt'

is_lm = False

if is_lm:
    output_video_path = f'output_lm_video_{idx}.mp4'
else:
    output_video_path = f'output_video_{idx}.mp4'


def draw_landmarks_on_image(detection_result, rgb_img, mode):
  if mode == 'image':
    annotated_image = np.copy(rgb_img)
    return torch.from_numpy(annotated_image)
  elif mode == 'skeleton':
    annotated_image = np.zeros_like(rgb_img)
  elif mode == 'image_and_skeleton':
    annotated_image = np.copy(rgb_img)

  # Loop through the detected hands to visualize.
  for idx in range(len(detection_result)):
    hand_landmarks = detection_result[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark[0], y=landmark[1], z=landmark[2]) for landmark in hand_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

  return annotated_image


# Thresholds for active probability and keypress probability
active_prob_threshold = 0.7
keypress_prob_threshold = 0.9

# Delay in frames
frame_delay = 3

# Load CSV files
first_csv = pd.read_csv(first_csv_path)
second_csv = pd.read_csv(second_csv_path)

# Video writer setup (read the first frame to get dimensions)
sample_frame_path = os.path.join(frame_folder, 'frame_0.jpg')
sample_frame = cv2.imread(sample_frame_path)
frame_height, frame_width = sample_frame.shape[:2]
fps = 5.0  # Set your desired frames per second

# Helper function to overlay text on frames
def add_text_to_frame(frame, text, position, font_scale=0.5, color=(255, 255, 255), thickness=2):
    return cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

# Initialize variables
last_key = None
last_prediction_text = "Prediction: None"  # Stores the last valid prediction text
frame_count = len(second_csv)

processed = []
gt = []
video = torch.load(lm_path, weights_only=True)
frames_list = []

for i in tqdm(range(40, 500)):
    # Read frame
    
    lm = video[i]
    img = np.zeros(shape=[360, 640, 3])
    img = draw_landmarks_on_image(
                lm.numpy(), 
                img,
                mode="skeleton"
            )
    img = np.array(img, np.uint8)
    if is_lm:
        frame = img
    else:
        frame_path = os.path.join(frame_folder, f'frame_{i}.jpg')
        frame = cv2.imread(frame_path)

    # print(img.shape)
    # print(np.unique(img))
    # print(np.unique(frame))
    # frame = img
    if frame is None:
        print(f"Frame {frame_path} not found. Skipping.")
        continue
    
    # Get row data for the current frame
    second_row = second_csv.iloc[i]
    frame_num = second_row['Start frame'] + 3
    active_prob = second_row['Active Prob']
    keypress = second_row['Key prediction']
    keypress_prob = second_row['Key Prob']

    # Check if we should update the "Prediction: key" text
    if active_prob > active_prob_threshold and keypress_prob > keypress_prob_threshold:
        last_prediction_text = f"Prediction: {keypress}"  # Update the last valid prediction text
        if keypress == 'dot':
            processed.append('.')
        elif keypress == 'comma': 
            processed.append(',')
        elif keypress == 'space':
            processed.append(' ')
        elif keypress == 'delete':
            if len(processed):
                processed.pop()
        else:
            processed.append(keypress)
    
    # Display the last valid prediction text if it exists
    color = (0, 255, 0) if keypress == last_key else (255, 255, 255)
    
    if last_key == 'dot':
        gt.append('.')
    elif last_key == 'comma': 
        gt.append(',')
    elif last_key == 'space':
        gt.append(' ')
    elif last_key == 'delete':
        if len(processed):
            gt.pop()
    else:
        gt.append(last_key)
    
    # frame = add_text_to_frame(frame, ''.join(gt), (frame_width - 300, 50), color=color)
    frame = add_text_to_frame(frame, remove_consecutive_letters(''.join(processed)), (50, 50), color=color)
    
    frame = add_text_to_frame(frame, last_prediction_text, (frame_width - 250, frame_height - 30), color=color)

    # Check for new keypress in first_csv with a 3-frame delay
    if i >= frame_delay and (frame_num - frame_delay) in first_csv['Frame'].values:
        delayed_row = first_csv[first_csv['Frame'] == frame_num - frame_delay]
        if not delayed_row.empty:
            new_key = delayed_row.iloc[0]['Key']
            if new_key != last_key:
                last_key = new_key  # Update only when a new key appears
    
    # Keep displaying the last key until a new key appears
    frame = add_text_to_frame(frame, f"Last Key: {last_key};", (200, frame_height - 30), color=color)
    
    # Display the frame number for debugging purposes
    frame = add_text_to_frame(frame, f"Frame: {frame_num - 3}", (50, frame_height - 30), color=color)

    # Convert BGR to RGB (cv2 uses BGR, torchvision expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to torch tensor and add to list
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1)
    frames_list.append(frame_tensor)

# Convert to torch tensor and add to list
frames_tensor = torch.stack(frames_list)
# Permute from [N, C, H, W] to [N, H, W, C]
frames_tensor = frames_tensor.permute(0, 2, 3, 1)
tvio.write_video(output_video_path, frames_tensor, fps=fps)

print("Video processing complete.")