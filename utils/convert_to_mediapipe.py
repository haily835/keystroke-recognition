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

data_dir = './datasets/angle-2/segments_dir'
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

class_dirs = sorted(os.listdir(data_dir))

destination_image_dir = f"./datasets/angle-2/landmarks"

if not os.path.exists(destination_image_dir):
    os.makedirs(destination_image_dir)

to_img = True
for class_dir in class_dirs:
    label = class_dir
    samples = os.listdir(f"{data_dir}/{class_dir}")
    class_samples = []
    for sample in tqdm(samples):
        frames = []
        jpgs = sorted(glob.glob(f"{data_dir}/{class_dir}/{sample}/*.jpg"))
        
        landmark_frames = []
        for jpg in jpgs:
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
                # print("Enough landmarks")
            
            if to_img:
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
                    
                    if not os.path.exists(f'{destination_image_dir}/{class_dir}'):
                        os.makedirs(f'{destination_image_dir}/{class_dir}')
                    
                    landmark_frames.append(torch.tensor(annotated_image))
                    # torchvision.io.image.write_jpeg(
                    #     input = torch.tensor(annotated_image).permute(2, 0, 1), # permute back to (c, h, w)
                    #     filename=f'{destination_image_dir}/{class_dir}/{sample}/{img_name}.jpg', 
                    #     quality=60
                    # )
            
        torchvision.io.video.write_video(
            f'{destination_image_dir}/{class_dir}/{sample}.mp4',
            torch.stack(landmark_frames),
            fps=1.0
        )
        # if (len(frames) != len(jpgs)):
        #     print("Medipipe can not identify all fingers")
        #     print(f"Skipping {data_dir}/{class_dir}/{sample}")
        # else:
        #     print(f"Success {data_dir}/{class_dir}/{sample}")
        if (len(frames) == len(jpgs)):
            class_samples.append(frames)
    
    class_samples = np.array(class_samples)
    print(f"Success: {len(class_samples)}. Total {len(samples)}")
    class_samples = torch.tensor(class_samples)

    torch.save(class_samples, f'{destination_image_dir}/{class_dir}.pt')