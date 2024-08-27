from typing import List, Tuple
import torch
import lightning as L
import pandas as pd
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import CombinedLoader
import torchvision.transforms.v2 as v2
import lightning as L
import pandas as pd
from lightning_utils.lm_dataset import BaseStreamDataset


def get_dataloader(
        frames_dir,
        labels_dir,
        landmarks_dir,
        videos,
        idle_gap=None,
        delay=10,
        batch_size=4,
        num_workers=4,
        shuffle=False):

    key_counts = pd.DataFrame()
    datasets = []
    datasets = [BaseStreamDataset.create_dataset(
            video_path=f"{frames_dir}/{video}",
            landmark_path=f"{landmarks_dir}/{video}.pt",
            label_path=f"{labels_dir}/{video}.csv",
            gap=idle_gap,
            delay=delay,
        ) for video in videos]
    
    key_counts['label'] = datasets[0].get_class_counts()['label']
    for video, ds in zip(videos, datasets):
        key_counts[video] = ds.get_class_counts()['count']

    merged = torch.utils.data.ConcatDataset(datasets)
    print('Key counts: \n', key_counts)
    print("Total samples: ", len(merged))

    loader = DataLoader(
        merged,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers,
        shuffle=shuffle
    )
    return loader


class LMKeyStreamModule(L.LightningDataModule):
    def __init__(self,
                 frames_dir: str,
                 landmarks_dir: str,
                 labels_dir: str,
                 train_videos: List[str] = [],
                 val_videos: List[str] = [],
                 test_videos: List[str] = [],
                 idle_gap: int = None,
                 delay: int = 10,
                 batch_size: int = 4,
                 num_workers: int = 4):
        """
        train_collate_fns: create multiple loaders for every collate functions and a loader without any collate function
        idle_gap=None: if None, the binary detect (idle or active segments) dataset will be used
        """
        super().__init__()

        self.train_loader = get_dataloader(frames_dir,
                                           labels_dir,
                                           landmarks_dir,
                                           videos=train_videos,
                                           idle_gap=idle_gap,
                                           delay=delay,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True) if len(train_videos) else None

        self.val_loader = get_dataloader(
            frames_dir,
            labels_dir,
            landmarks_dir,
            videos=val_videos,
            idle_gap=idle_gap,
            delay=delay,
            batch_size=batch_size,
            num_workers=num_workers,
        ) if len(val_videos) else None

        self.test_loader = get_dataloader(
            frames_dir,
            labels_dir,
            landmarks_dir,
            videos=test_videos,
            idle_gap=idle_gap,
            delay=delay,
            batch_size=batch_size,
            num_workers=num_workers,
        ) if len(test_videos) else None

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
