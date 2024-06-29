import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import pathlib
import torch
# from GesRec.models.resnet import resnet101
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
import lightning as L
from lightning.pytorch.loggers import CSVLogger

class KeyClf(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = resnet101(sample_size=224,
                 sample_duration=16,
                 shortcut_type='A',
                 num_classes=30)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        videos, targets = batch
        outputs = self.model(videos)
        loss = self.loss_fn(outputs, targets.long())

        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)
    

class KeyStrokeClsDataset(Dataset):
    def __init__(self, dataset_root_path, mode):
        self.dataset_root_path = pathlib.Path(dataset_root_path)
        self.all_video_file_paths =  list(self.dataset_root_path.glob(f"{mode}/*/*.mp4"))
        self.class_labels = sorted({str(path).split("/")[-2] for path in self.all_video_file_paths})
        self.label2id = {label: i for i, label in enumerate(self.class_labels)}
        self.id2label  = {i: label for label, i in self.label2id.items()}

    def __len__(self):
        return len(self.all_video_file_paths)

    def __getitem__(self, idx):
        file_path = self.all_video_file_paths[idx]
        vframes, _, _ = read_video(file_path, pts_unit='sec')
        label = str(file_path).split("/")[-2]

        # permute to (num_frames, num_channels, height, width)
        vframes = vframes.permute(3, 0, 1, 2).float() / 255.0
    
        return vframes, self.label2id[label]
