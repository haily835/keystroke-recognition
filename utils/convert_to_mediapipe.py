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

src = f'./datasets/shortvideo-1/raw_frames/video_0'
dest = './datasets/shortvideo-1/landmarks/video_0'

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)


if not os.path.exists(dest):
    os.makedirs(dest)


jpgs = sorted(glob.glob(f"{src}/*.png"))

to_img = True

frames = []
for jpg in tqdm(jpgs):
    img = torchvision.io.image.read_image(jpg)

    img = img.permute(1, 2, 0).numpy()

    pil_img = Image.fromarray(img)

    data = np.asarray(pil_img)
    media_pipe_img = mp.Image(
        image_format=mp.ImageFormat.SRGB, 
        data=data # mediapipe does not work with normal torch 
    )

    detection_result = detector.detect(media_pipe_img)
    hand_landmarks_list = detection_result.hand_landmarks

    coords = [[], []] # coordinates of 21 points of 2 hands
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        for landmark in hand_landmarks:
            coords[idx].append((landmark.x, landmark.y, landmark.z))
        

    if len(coords[1]) == 21 and len(coords[0]) == 21:
        frames.append(coords)

    if not to_img: continue
    
    annotated_image = np.copy(data)
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

    img_name = jpg.split('/')[-1]
   
    torchvision.io.image.write_png(
        torch.tensor(annotated_image).permute(2, 0, 1),
        f'{dest}/{img_name}',
        compression_level=9,
    )

print(f"Sucessed {len(frames)} in {len(jpgs)}")
torch.save(torch.stack(frames), f'{dest}/coordinates.pt')