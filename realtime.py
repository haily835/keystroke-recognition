import asyncio
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
import socket
import os
import cv2
import threading

host = '127.0.0.1'  # Localhost
port = 65432  # Same port as sender

id2Label = ['[i]', 'BackSpace', ',', '[s]', '.', 
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
            'y', 'z']
label2Id  = {label: i for i, label in enumerate(id2Label)}
labels_dir = './labels'
videos_dir = './datasets/raw_frames'

NUM_WORKERS = 0

#### RESNET 3D #### 
def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4) ).zero_()
    
    if isinstance(out.data, torch.cuda.FloatStorage): zero_pads = zero_pads.cuda()
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
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        
        
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
                    nn.Conv3d(self.inplanes,planes * block.expansion,kernel_size=1,stride=stride,bias=False), 
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

class KeyClf(L.LightningModule):
    def __init__(self, img_size, num_classes, learning_rate, weights):
        super().__init__()
        self.model = resnet101(sample_size=img_size, 
                               sample_duration=8,
                               shortcut_type='B', 
                               num_classes=num_classes)
        
        self.loss_fn = torch.nn.CrossEntropyLoss(torch.tensor(weights).float())
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = learning_rate
        self.transforms = v2.Compose([
            v2.CenterCrop(img_size),
            v2.ToDtype(torch.float32, scale=True),
        ])
        
        self.test_preds = []
        self.test_targets = []
        self.save_hyperparameters()
    
    def test_step(self, batch):
        videos, targets = batch
        videos = self.transforms(videos)
        videos = videos.permute(0, 2, 1, 3, 4)
        preds = self.model(videos)

        pred_ids = torch.argmax(self.model(videos), dim=1).squeeze()
        pred_labels = [id2Label[_id] for _id in pred_ids]
        self.test_preds += pred_labels
        self.test_targets += [id2Label[_id] for _id in targets]
        
        loss = self.loss_fn(preds, targets.long())
        self.log_dict({'test_acc': self.accuracy(preds, targets), 'test_loss': loss})
    
    def on_test_end(self):
        print(classification_report(self.test_targets, self.test_preds))
        
    def training_step(self, batch):
        videos, targets = batch
        videos = self.transforms(videos)
        videos = videos.permute(0, 2, 1, 3, 4)
        preds = self.model(videos)
        loss = self.loss_fn(preds, targets.long())
        self.log_dict({"train_loss": loss, "train_acc": self.accuracy(preds, targets)})
        return loss

    def validation_step(self, batch):
        videos, targets = batch
        videos = self.transforms(videos)
        videos = videos.permute(0, 2, 1, 3, 4)
        preds = self.model(videos)
        loss = self.loss_fn(preds, targets.long())
        self.log_dict({"val_loss": loss, "val_acc": self.accuracy(preds, targets)})
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
trained_model = KeyClf.load_from_checkpoint("ckpts/epoch=7-step=34979-full.ckpt")
trained_model.freeze()

image_folder = './socket'  # Folder to store received images
def receive_images(conn):
    i = 0
    try:
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        while True:
            image_size_bytes = conn.recv(8)
            if not image_size_bytes:
                print("Did not receive image size.")
                break
            image_size = int.from_bytes(image_size_bytes, 'big')
            received_data = b''
            while len(received_data) < image_size:
                packet = conn.recv(min(image_size - len(received_data), 4096))
                if not packet:
                    break
                received_data += packet

            if len(received_data) == image_size:
                image_array = np.frombuffer(received_data, dtype=np.uint8)
                frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                if frame is not None:
                    # print("Frame received and saved.")
                    image_path = os.path.join(image_folder, f'frame{i}.jpg')
                    cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    i += 1
                else:
                    print("Failed to decode frame.")
            else:
                print("Received data size does not match the expected size.")

    except Exception as e:
        print(f"Error in receiving images: {e}")


def predict_images():
    window = []
    window_size = 8
    i = 0
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    try:
        while True:
            # Check if there are enough images for prediction
            image_files = os.listdir(image_folder)
            image_path = os.path.join(image_folder, f'frame{i}.jpg')
            # no frame yet
            if not len(image_files):
                time.sleep(1)
            elif len(window) < window_size:
                image = torchvision.io.read_image(image_path)
                window.append(image)
                i += 1
            elif len(window) == window_size: 
                # Load and predict on the batch of images
                frames = torch.stack(window)
                frames = trained_model.transforms(frames)
                frames = frames.permute(1, 0, 2, 3)
                out = F.softmax(trained_model.model(frames.unsqueeze(0)))[0]
                _id = torch.argmax(out)
                label = id2Label[_id]
                print(f"{i - 7};{label};{out[_id]}")
            
                image = torchvision.io.read_image(image_path)
                window.append(image)
                window = window[1:]
                i += 1
    except Exception as e:
        print(f"Error in predicting images: {e}")

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.listen(1)
        print(f"Listening on {host}:{port}")
        conn, addr = s.accept()
        print(f"Connected by {addr}")

        # Start receiving images in a separate thread
        threading.Thread(target=receive_images, args=(conn,)).start()
        # Start predicting images in the main thread
        time.sleep(1)
        predict_images()

    except Exception as e:
        print(f"Error: {e}")

    finally:
        conn.close()
        s.close()

if __name__ == "__main__":
    import time
    s = time.perf_counter()
    main()
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")