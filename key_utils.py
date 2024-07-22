from keras import preprocessing
import torch
import torch.nn as nn
import lightning as L
import pandas as pd
import torchvision
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import pathlib
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_video
import lightning as L
from lightning.pytorch.loggers import CSVLogger
import torchmetrics
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torchvision.transforms import CenterCrop, v2
from datetime import datetime
import csv
import glob
import os


id2label = ['[i]', 'BackSpace', ',', '[s]', '.',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x',
            'y', 'z']
label2id = {label: i for i, label in enumerate(id2label)}


class KeyDataset(torch.utils.data.Dataset):
    def __init__(self, video_name, labels_dir, videos_dir,
                 f_before=2, f_after=2, gap=2, total_window=5,
                 color_channel_last=True, landmark_format=False,
                 preprocess=None):
        segments = []
        # Infer idle frames.
        self.labels_dir = labels_dir
        self.videos_dir = videos_dir
        self.video_name = video_name
        df = pd.read_csv(f'{self.labels_dir}/{video_name}.csv')
        for index, row in df.iterrows():
            key_frame = int(row['Frame'])  # Frame number where key was pressed
            key_value = row['Key']  # Key pressed
            if key_value not in id2label:
                key_value = '[s]'

            is_idle_before = False
            if index == 0:
                pos_start = max(key_frame - f_before, 0)
                pos_end = key_frame + f_after
                neg_start = 0
                neg_end = pos_start - gap
                is_idle_before = True
            else:
                prev_key_frame = df.iloc[index - 1]['Frame']
                pos_start = max(key_frame - f_before, 0)
                pos_end = key_frame + f_after
                prev_pos_end = prev_key_frame + f_after
                if (pos_start - prev_pos_end) - 1 >= (f_after + f_before + 1 + gap * 2):
                    neg_start = prev_pos_end + gap
                    neg_end = pos_start - gap
                    is_idle_before = True

            # Negative class video segments before
            if is_idle_before:
                j = neg_start
                while (j + total_window - 1) <= neg_end:
                    segments.append(([j, j + total_window - 1], "[i]"))
                    j += total_window

            # Current video with keystroke
            segments.append(([pos_start, pos_end], key_value))

        if landmark_format:
            self.landmarks = torch.load(f'{videos_dir}/{video_name}.pt')
            # print(self.landmarks.shape)
        self.landmark_format = landmark_format
        self.color_channel_last = color_channel_last
        self.preprocess = preprocess
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        (start, end), label = self.segments[idx]
        if self.landmark_format:
            return self.landmarks[start:end+1], label2id[label]
        else:
            frames = []
            for i in range(start, end + 1):
                image = torchvision.io.read_image(
                    f"{self.videos_dir}/{self.video_name}/frame_{i}.jpg")
                frames.append(image)

            frames = torch.stack(frames)
            if self.color_channel_last:
                frames = frames.permute(0, 2, 3, 1)

            if self.preprocess:
                frames = self.preprocess(frames)

            return frames, label2id[label]

    def get_class_counts(self):
        labels = [segment[1] for segment in self.segments]
        unique_elements, counts = np.unique(labels, return_counts=True)
        occurrences = dict(zip(unique_elements, counts))
        weights = np.zeros(len(id2label))
        for label, count in occurrences.items():
            weights[label2id[label]] = count
        return weights


class KeyDataModule(L.LightningDataModule):
    def __init__(self, labels_dir, videos_dir,
                 batch_size=4,
                 landmark_format=False,
                 color_channel_last=False,
                 num_workers=4,
                 preprocess=None,
                 train_vids=[
                     'video_1', 'video_2', 'video_3', 'video_4', 'video_5',
                     'video_6', 'video_7', 'video_8', 'video_9', 'video_10',
                     'video_11', 'video_12', 'video_13', 'video_14', 'video_15',
                     'video_16', 'video_17', 'video_18', 'video_19',
                     'video_21', 'video_22', 'video_23', 'video_24', 'video_25',
                     'video_26', 'video_27', 'video_28', 'video_29', 'video_30'],
                 val_vids=['video_31', 'video_32', 'video_33'],
                 test_vids=['video_34', 'video_35', 'video_36']):
        super().__init__()
        self.batch_size = batch_size
        self.labels_dir = labels_dir
        self.videos_dir = videos_dir
        self.train_vids = train_vids
        self.val_vids = val_vids
        self.test_vids = test_vids
        self.num_workers = num_workers
        self.landmark_format = landmark_format
        self.color_channel_last = color_channel_last
        self.preprocess = preprocess

        train_datasets = [KeyDataset(video_name,
                                     self.labels_dir,
                                     self.videos_dir,
                                     landmark_format=landmark_format,
                                     color_channel_last=landmark_format,
                                     preprocess=self.preprocess)
                          for video_name in self.train_vids
                          ]
        train_counts = np.array([d.get_class_counts()
                                for d in train_datasets]).sum(axis=0)
        train_total_samples = np.array(
            [len(d) for d in train_datasets]).sum(axis=0)
        train_weights = train_counts / (train_total_samples * len(id2label))
        self.train_weights = train_weights

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train = torch.utils.data.ConcatDataset(
                [KeyDataset(video_name, self.labels_dir, self.videos_dir,
                            landmark_format=self.landmark_format,
                            color_channel_last=self.landmark_format,
                            preprocess=self.preprocess)
                 for video_name in self.train_vids
                 ]
            )
            self.val = torch.utils.data.ConcatDataset(
                [KeyDataset(video_name, self.labels_dir, self.videos_dir, landmark_format=self.landmark_format, preprocess=self.preprocess,
                            color_channel_last=self.landmark_format) for video_name in self.val_vids]
            )

            print(f"Train: {len(self.train)}; Val: {len(self.val)};")
        if stage == "test":
            self.test = torch.utils.data.ConcatDataset(
                [KeyDataset(video_name, self.labels_dir, self.videos_dir,  landmark_format=self.landmark_format, preprocess=self.preprocess,
                            color_channel_last=self.landmark_format)
                 for video_name in self.test_vids]
            )
            print(f"Test: {len(self.test)};")

    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=False)


class KeyClf(L.LightningModule):
    def __init__(self, weights, 
                 num_classes=len(id2label), 
                 learning_rate=0.01):
        
        super().__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(torch.tensor(weights).float())
        self.accuracy = torchmetrics.Accuracy(
            task="multiclass", 
            num_classes=num_classes)
        self.lr = learning_rate
        self.test_preds = []
        self.test_targets = []

        self.save_hyperparameters()

    def common_step(self, batch):
        videos, targets = batch
        preds = self.model(videos)
        pred_ids = torch.argmax(preds, dim=1)
        loss = self.loss_fn(preds, targets.long())
        return loss, pred_ids

    def test_step(self, batch):
        _, targets = batch
        loss, pred_ids = self.common_step(batch)
        pred_labels = [id2label[_id] for _id in pred_ids]
        self.test_preds += pred_labels
        self.test_targets += [id2label[_id] for _id in targets]
        self.log('test_loss', loss, sync_dist=True, prog_bar=True)

    def on_test_end(self):
        print(classification_report(self.test_targets, self.test_preds))

    def training_step(self, batch):
        loss, _ = self.common_step(batch)
        self.log('tran_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss, _ = self.common_step(batch)
        self.log('val_loss', loss, sync_dist=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)


##################### MODELS ###########################
##################### MODELS ###########################
##################### MODELS ###########################
#### RESNET 3D ####
def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(
        0), planes - out.size(1), out.size(2), out.size(3), out.size(4)).zero_()

    if isinstance(out.data, torch.cuda.FloatStorage):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(
            planes, planes,
            kernel_size=3, stride=stride, padding=1,
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, sample_size, sample_duration, shortcut_type='B', num_classes=400):
        """
        block: basic block or bottle neck
        layers: define Resnet architecture 34, 101, 152 etc
        sample size: image size
        shortcut_type: 'A' or 'B'
        num_classes: ...
        """
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(
            1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2)

        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet10(**kwargs): return ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
def resnet18(**kwargs): return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
def resnet34(**kwargs): return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
def resnet50(**kwargs): return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
def resnet101(**kwargs): return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
def resnet152(**kwargs): return ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
def resnet200(**kwargs): return ResNet(Bottleneck, [3, 24, 36, 3], **kwargs)
