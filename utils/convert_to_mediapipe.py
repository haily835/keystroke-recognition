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

import argparse


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)
detector = vision.HandLandmarker.create_from_options(options)


def process_image(image_path):
    img = torchvision.io.image.read_image(image_path)
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

# Usage in the main loop.
# frames = []
# for i in range(len(jpgs)):
#     image_path = f"{src}/frame_{i}.jpg"
#     result = process_image(image_path, detector)
#     if result is not None:
#         frames.append(result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert raw frames to landmarks')
    parser.add_argument('--src_dir', type=str, required=True, help='Source directory containing raw frames')
    parser.add_argument('--dest_dir', type=str, required=True, help='Destination directory for landmarks')
    args = parser.parse_args()

    src_dir = args.src_dir
    dest_dir = args.dest_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for video in range(12):
        video_name = f'video_{video}'
        src = f'{src_dir}/{video_name}'
        dest = f'{dest_dir}/{video_name}.pt'

        jpgs = sorted(glob.glob(f"{src}/*.jpg"))

        to_img = False

        frames = []
        last_succeed = None
        for i in range(len(jpgs)):
            img_path = f"{src}/frame_{i}.jpg"
            result = process_image(img_path)
            last_succeed = result
            if result is not None:
                frames.append(result)
            else:
                frames.append(last_succeed)
                print(f'Mediapipe failed at {i}')

        print(f"Sucessed {len(frames)} in {len(jpgs)}")
        torch.save(torch.stack(frames), dest)