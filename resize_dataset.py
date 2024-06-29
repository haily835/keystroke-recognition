import pathlib
import torch
# from GesRec.models.resnet import resnet101
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
import lightning as L
import torchvision.transforms.functional as TF
import os
from lightning.pytorch.loggers import CSVLogger
import cv2
from tqdm import tqdm
from PIL import Image
import numpy as np
mode = 'train'

fps = 30
width, height = 112, 112
source = pathlib.Path(f"./datasets/key_clf_data_320_320")
all_video_file_paths =  list(source.glob(f"{mode}/*/*.mp4"))

target = f'./datasets/key_clf_data_112_112'

for video_path in tqdm(all_video_file_paths):
    label = str(video_path).split("/")[-2]
    video_name = str(video_path).split("/")[-1]
    vframes, _, _ = read_video(video_path, pts_unit='sec')
  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    os.makedirs(f"{target}/{mode}/{label}", exist_ok=True)

    output_video_path = f"{target}/{mode}/{label}/{video_name}"

    out = cv2.VideoWriter(output_video_path, fourcc, 30, (width, height), isColor=True)

    for frame in vframes.numpy(): 
        # Convert frame to PIL Image
        img = Image.fromarray(frame, mode="RGB")
        img = img.resize((width, height))  # Resize if necessary

        # Convert PIL Image back to numpy array
        frame_np = np.array(img)
        
        img_cv2 = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(img_cv2)

    out.release()