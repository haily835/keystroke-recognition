import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import glob
import os
from tqdm import tqdm
import torch
import torchvision.transforms.functional
import torchvision.transforms.v2
import torchvision


src = f'./datasets/angle-2/raw_frames'
dest = './datasets/angle-2/raw_frames_224'
if not os.path.exists(dest):
  os.mkdir(dest)
videos = [
  'video_0', 'video_1', 'video_2', 'video_3', 'video_4', 'video_5', 'video_6', 'video_7', 'video_8', 'video_9', 'video_10',
  'video_11', 'video_12', 'video_13', 'video_14']

for video in videos:
  print('video: ', video)

  image_paths = glob.glob(f"{src}/{video}/*.jpg")

  for path in tqdm(image_paths):
    image = torchvision.io.read_image(path)
    h, w = image.shape[-2], image.shape[-1]
    image = torchvision.transforms.functional.resized_crop(
        image,
        top=3*h//5,
        left=0,
        height=2*h//5,
        width=w,
        size=(224, 224),
        antialias=True
    )
    image_name = path.split('/')[-1]

    if not os.path.exists(f"{dest}/{video}"):
      os.mkdir(f"{dest}/{video}")
    torchvision.io.write_jpeg(image, f"{dest}/{video}/{image_name}", quality=60)