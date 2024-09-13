import torch
import lightning as L
import pandas as pd
import torch.utils
import torch.utils.data
from torch.utils.data import DataLoader
import lightning as L
import pandas as pd
import glob
from lightning_utils.dataset import clf_id2label, clf_id2label
import numpy as np

class LMDataset(torch.utils.data.Dataset):
    def __init__(self, video_ids = range(10), id2label=clf_id2label, label2id=clf_id2label):
        all_samples = []
        all_labels = []
        for video_id in video_ids:
            video_paths = []
            label_paths = []

            video_paths.append(glob.glob(f'./datasets/topview-2/landmarks_aug/video_{video_id}_orgi.pt')[0])
            video_paths.append(glob.glob(f'./datasets/topview-2/landmarks_aug/video_{video_id}_rotate.pt')[0])
            video_paths.append(glob.glob(f'./datasets/topview-2/landmarks_aug/video_{video_id}_zoom.pt')[0])
            label_paths.append(glob.glob(f'./datasets/topview-2/landmarks_aug/video_{video_id}_orgi_label.pt')[0])
            label_paths.append(glob.glob(f'./datasets/topview-2/landmarks_aug/video_{video_id}_rotate_label.pt')[0])
            label_paths.append(glob.glob(f'./datasets/topview-2/landmarks_aug/video_{video_id}_zoom_label.pt')[0])

            for vp, lp in zip(video_paths, label_paths):
                all_samples.append(torch.load(vp, weights_only=True))
                all_labels.append(torch.load(lp, weights_only=True))
        
        self.all_samples = torch.concat(all_samples, dim=0)
        self.all_labels = torch.concat(all_labels, dim=0)
        self.id2label = id2label
        self.label2id = label2id
    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        return self.all_samples[idx], self.all_labels[idx]

    def get_class_counts(self):
        labels = self.all_labels.numpy()

        counts = []
        for label in self.id2label:
            count = np.sum(labels == label)
            counts.append(count)

        df = pd.DataFrame({'label': self.id2label, 'count': counts})
        return df

class LMModule(L.LightningDataModule):
    def __init__(self,batch_size: int = 4,num_workers: int = 4):
        super().__init__()
    
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_ds = LMDataset(video_ids=range(8))
        val_ds = LMDataset(video_ids=[8, 9])
        
        self.train_loader =  DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers,
            shuffle=True
        )

        self.val_loader =  DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=num_workers,
            shuffle=True
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
