import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
import pathlib
import torch
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


LOCAL = True

# Training hyperparameters
IMG_SIZE = 320
FRAMES_PER_VIDEO = 8
NUM_CLASSES = 30
LEARNING_RATE = 0.001
BATCH_SIZE = 4
MAX_EPOCHS = 2
MAX_TIME = "00:11:00:00"

# Dataset
LOCAL_DATA_DIR = f"./datasets/key_clf_data_{IMG_SIZE}_{IMG_SIZE}"
KAGGLE_DATA_DIR = f"/kaggle/input/key-clf/key_clf_data_{IMG_SIZE}_{IMG_SIZE}"

NUM_WORKERS = 4

FAST_DEV_RUN = False
CHECKPOINT_DIR = "resnet/"

# Compute related
ACCELERATOR = "cpu"
DEVICES = [0,1]

id2Label = ['BackSpace', 'Comma', 'Space', 'Stop', 
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 
            'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 
            'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
            'y', 'z']

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


####### 
class KeyClf(L.LightningModule):
    def __init__(self, img_size, frames_per_video, num_classes, learning_rate, weights):
        super().__init__()
        self.model = resnet101(sample_size=img_size, sample_duration=frames_per_video,
                               shortcut_type='B', num_classes=num_classes)
        
        self.loss_fn = torch.nn.CrossEntropyLoss(torch.tensor(weights))
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.lr = learning_rate
        self.test_y = []
        self.test_pred = []

    def test_step(self, batch):
        videos, targets = batch
        preds = self.model(videos)
        self.test_pred.append(preds)
        self.test_y.append(targets)

        loss = self.loss_fn(preds, targets.long())
        test_acc = self.accuracy(preds, targets)
        self.log_dict({'test_acc': test_acc, 'test_loss': loss})


    def on_test_end(self) -> None:
        preds = torch.cat(self.test_pred)
        targets = torch.cat(self.test_y)
        acc = self.accuracy(preds, targets)
        print('acc: ', acc)
        print("target", targets[:5])
        print("preds", torch.argmax(preds[:5], 1))

        print(classification_report(targets.numpy(), torch.argmax(preds, 1).numpy()))
        cm = confusion_matrix(targets.numpy(), torch.argmax(preds, 1).numpy())
        plt.figure(figsize=(16, 12))
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=id2Label, yticklabels=id2Label)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig('result.png')

    
    def training_step(self, batch):
        videos, targets = batch
        preds = self.model(videos)
        loss = self.loss_fn(preds, targets.long())
        self.log_dict({ "train_loss": loss, "train_acc": self.accuracy(preds, targets)}, 
                      on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        videos, targets = batch
        preds = self.model(videos)
        loss = self.loss_fn(preds, targets.long())
        self.log_dict({ "val_loss": loss, "val_acc": self.accuracy(preds, targets)}, 
                      on_step=False, on_epoch=True, prog_bar=True,)
        return loss
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
    
class KeyClsDataset(Dataset):
    def __init__(self, data_dir, mode):
        self.dataset_root_path = pathlib.Path(data_dir)
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

class KeyClsDataModule(L.LightningDataModule):
    def __init__(self, batch_size, data_dir):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_root_path = pathlib.Path(data_dir)
        self.data_dir = data_dir
        weights = {}
        probs = {'Letter': id2Label}
        
        for split in ['train', 'val', 'test']:
            video_paths =  list(self.dataset_root_path.glob(f"{split}/*/*.mp4"))

            class_labels = sorted({str(path).split("/")[-2] for path in video_paths})

            total = len(video_paths)
            weights[split] = []
            probs[split] = []
            for label in class_labels:
                samples = len(list(self.dataset_root_path.glob(f"{split}/{label}/*.mp4")))
                probs[split].append(samples)
                weights[split].append(total / (NUM_CLASSES * samples))
            
        self.weights = weights
        self.probs = probs
   
    def train_dataloader(self):
        train_dataset = KeyClsDataset(self.data_dir, 'train')
        print("Train dataset:", len(train_dataset))
        return DataLoader(train_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS, persistent_workers=True)
    
    def val_dataloader(self):
        val_dataset = KeyClsDataset(self.data_dir, 'val')
        print("Val dataset:", len(val_dataset))
        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS, persistent_workers=True)
    
    def test_dataloader(self):
        test_dataset = KeyClsDataset(self.data_dir, 'test')
        print("Test dataset:", len(test_dataset))
        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=NUM_WORKERS, persistent_workers=True)


if __name__ == '__main__':
    dm = KeyClsDataModule(batch_size=BATCH_SIZE, 
                          data_dir=LOCAL_DATA_DIR if LOCAL else KAGGLE_DATA_DIR)
    print(pd.DataFrame(dm.probs))
    model = KeyClf(img_size=IMG_SIZE, frames_per_video=FRAMES_PER_VIDEO, 
                   num_classes=NUM_CLASSES, learning_rate=LEARNING_RATE, 
                   weights = dm.weights['train'])
    
    logger = CSVLogger("logs", name=f"resnet", flush_logs_every_n_steps=1)
    
    if LOCAL:
        trainer = L.Trainer(
                    max_time=MAX_TIME,
                    callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
                    fast_dev_run=True,
                    logger=logger,
                    accelerator="cpu"
                )
    else: 
        trainer = L.Trainer(
                deterministic=False,
                devices=DEVICES,
                max_time=MAX_TIME,
                callbacks=[EarlyStopping(monitor="val_loss", patience=10)],
                fast_dev_run=FAST_DEV_RUN,
                logger=logger,
                accelerator=ACCELERATOR
            )
    
    trainer.fit(model, dm)
    trainer.test(model, dm)

    
       