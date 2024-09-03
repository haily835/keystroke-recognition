# Mediapip
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import glob
import torchvision
import torchvision.transforms.functional
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import rotate

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


def process_image(image_path, rotate_deg=0):
    img = torchvision.io.image.read_image(image_path)
    img = rotate(img, rotate_deg)
    img = img.permute(1, 2, 0).numpy()
    pil_img = Image.fromarray(img)
    data = np.asarray(pil_img)
    media_pipe_img = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=data
    )

    detection_result = detector.detect(media_pipe_img)
    hand_landmarks_list = detection_result.hand_landmarks

    coords = [[], []]  # coordinates of 21 points of 2 hands
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        for landmark in hand_landmarks:
            coords[idx].append((landmark.x, landmark.y, landmark.z))

    if len(coords[1]) == 21 and len(coords[0]) == 21:
        return torch.tensor(coords)
    return None

# Usage in the main loop:
# frames = []
# for i in range(len(jpgs)):
#     image_path = f"{src}/frame_{i}.jpg"
#     result = process_image(image_path, detector)
#     if result is not None:
#         frames.append(result)

rotate_deg = 5
if __name__ == '__main__':
    for video in range(5):
        video_name = f'video_{video}'
        print(video_name)
        src = f'datasets/video-2/raw_frames/{video_name}'
        dest = f'datasets/video-2/landmarks/{video_name}_d{rotate_deg}.pt'

        if not os.path.exists('datasets/video-2/landmarks'):
            os.makedirs('datasets/video-2/landmarks')

        jpgs = sorted(glob.glob(f"{src}/*.jpg"))

        to_img = False

        frames = []
        for i in tqdm(range(len(jpgs))):
            img_path = f"{src}/frame_{i}.jpg"
            result = process_image(img_path, )
            
            if result is not None:
                frames.append(result)
        
        print(f"Sucessed {len(frames)} in {len(jpgs)}")
        torch.save(torch.stack(frames), dest) 