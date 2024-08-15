import torch.utils
import torch.utils.data
import torchvision
import lightning as L
import pandas as pd
import torchvision.transforms.v2
from pytorchvideo.transforms.mix import MixUp
from lightning_utils.dataset import clf_id2label

def scale_fn(data):
    videos, labels = zip(*data)
    videos = torch.stack(videos)
    labels = torch.tensor(labels).long()

    videos = videos.permute(0, 2, 1, 3, 4)
    videos = (videos / 255.0).float()
    
    return videos, labels

mix_up = MixUp(alpha=0.4, num_classes=len(clf_id2label), label_smoothing=0.2)
def mix_up_fn(data):
    videos, labels = zip(*data)
    labels = torch.tensor(labels).long()
    videos = torch.stack(videos)
    new_videos, new_labels = mix_up(videos, labels)

    new_videos = new_videos.permute(0, 2, 1, 3, 4)
    new_videos = (new_videos / 255.0).float()
    
    return new_videos, new_labels


f_transforms = torchvision.transforms.v2.Compose([
    torchvision.transforms.v2.ColorJitter(brightness=.5, contrast=.5),
    torchvision.transforms.v2.RandomRotation(degrees=10)
])

def color_zoom_fn(data):
    videos, labels = zip(*data)
    videos = torch.stack(videos)
    labels = torch.tensor(labels).long()

    videos = f_transforms(videos)
    videos = videos.permute(0, 2, 1, 3, 4)
    videos = (videos / 255.0).float()
    return videos, labels