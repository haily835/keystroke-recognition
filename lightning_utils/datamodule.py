import torch
import lightning as L
import pandas as pd
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
from lightning.pytorch.utilities import CombinedLoader

import lightning as L
import pandas as pd
from lightning_utils.dataset import BaseStreamDataset
from utils.collate_fn import *


def get_dataloader(
        frames_dir,
        labels_dir,
        videos,
        collate_fn=None,
        idle_gap=None,
        batch_size=4,
        num_workers=4,
        transforms=None,
        shuffle=False):

    key_counts = pd.DataFrame()
    datasets = [BaseStreamDataset.create_dataset(
        video_path=f"{frames_dir}/{video}",
        label_path=f"{labels_dir}/{video}.csv",
        gap=idle_gap,
        transforms=transforms
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
        collate_fn=collate_fn,
        shuffle=shuffle
    )
    return loader


class KeyStreamModule(L.LightningDataModule):
    def __init__(self,
                 frames_dir,
                 labels_dir,
                 train_videos=[],
                 val_videos=[],
                 test_videos=[],
                 idle_gap=None,
                 batch_size=4,
                 num_workers=4,
                 train_collate_fns=[],
                 val_collate_fn=None,
                 test_collate_fn=None):
        """
        train_collate_fns: create multiple loaders for every collate functions and a loader without any collate function
        idle_gap=None: if None, the binary detect (idle or active segments) dataset will be used
        """
        super().__init__()
        train_batch_size = batch_size // (len(train_collate_fns) + 1)
        self.train_loader = [
            get_dataloader(frames_dir,
                           labels_dir,
                           videos=train_videos,
                           idle_gap=idle_gap,
                           batch_size=train_batch_size,
                           num_workers=num_workers,
                           collate_fn=scale_fn,
                           shuffle=True),
            *[get_dataloader(frames_dir,
                             labels_dir,
                             videos=train_videos,
                             idle_gap=idle_gap,
                             batch_size=train_batch_size,
                             num_workers=num_workers,
                             collate_fn=eval(train_collate_fn),
                             shuffle=True) for train_collate_fn in train_collate_fns]

        ] if len(train_videos) else None
        self.val_loader = get_dataloader(
            frames_dir,
            labels_dir,
            videos=val_videos,
            idle_gap=idle_gap,
            batch_size=batch_size,
            collate_fn=eval(val_collate_fn),
            num_workers=num_workers,
        ) if len(val_videos) else None
        self.test_loader = get_dataloader(
            frames_dir,
            labels_dir,
            videos=test_videos,
            idle_gap=idle_gap,
            batch_size=batch_size,
            collate_fn=eval(test_collate_fn),
            num_workers=num_workers,
        ) if len(test_videos) else None

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
