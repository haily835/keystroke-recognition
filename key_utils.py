import torch
import lightning as L
import pandas as pd
import torchvision
import numpy as np
from torch.utils.data import DataLoader
import lightning as L
import torchmetrics
from sklearn.metrics import classification_report
import pandas as pd
from einops import rearrange
import os
import torchvision.transforms.functional
import torchvision.transforms.v2
import shutil
import glob
from sklearn.model_selection import train_test_split
from collections import Counter

DELAY = 10

# id and label of all samples
id2label = ['comma', 'dot', 'BackSpace', 'idle', 'space', 
            'a', 'b', 'c', 'd', 'e', 'f', 
            'g', 'h', 'i', 'j', 'k', 'l', 
            'm', 'n', 'o', 'p', 'q', 'r', 
            's', 't', 'u', 'v', 'w', 'x', 
            'y', 'z']

label2id = {label: i for i, label in enumerate(id2label)}

class KeyStreamDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 video_name, data_dir,
                 f_before=2, f_after=2, gap=2,
                 rearrange_str = None,
                 gray_scale=False,
                 transforms=None,
                 crop=[]
                 ):
        """
        top: int, left: int, height: int, width: int
        """
        segments = []
        
        self.video_name = video_name
        self.data_dir = data_dir
        self.labels_dir = f"{data_dir}/labels"
        self.videos_dir = f"{data_dir}/raw_frames"
        self.rearrange_str = rearrange_str
        self.transforms = transforms
        self.gray_scale = gray_scale
        self.crop = crop

        df = pd.read_csv(f"{self.labels_dir}/{video_name}.csv")
        total_window = f_before + f_after + 1

        for index, row in df.iterrows():
            key_frame = int(row['Frame']) + DELAY  # Frame number where key was pressed
            key_value = row['Key']  # Key pressed
            if key_value not in id2label:
                key_value = '[i]'
            
            pos_start, pos_end = max(key_frame - f_before, 0), key_frame + f_after
            
            # Infer idle frames.
            is_idle_before = False
            if index == 0:
                neg_start, neg_end = 0, pos_start - gap
                is_idle_before = True
            else:
                prev_key_frame = df.iloc[index - 1]['Frame']
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
        
        self.segments = segments

    def __len__(self):
        return len(self.segments)

    def _get_frames(self, start, end):
        frames = []
        for i in range(start, end + 1):
            image = torchvision.io.read_image(f"{self.videos_dir}/{self.video_name}/frame_{i}.jpg")
            if len(self.crop):
                image = torchvision.transforms.functional.resized_crop(image, 720, 0, 560, 720, (224, 224))
            frames.append(image)
        frames = torch.stack(frames)

        if self.transforms:
            frames = self.transforms(frames)

        return frames

    def __getitem__(self, idx):
        (start, end), label = self.segments[idx]
        frames = self._get_frames(start, end)
        
        if self.rearrange_str:
            frames = rearrange(frames, self.rearrange_str)
        
        frames = frames / 255.0
        return frames.float(), label2id[label]

    def _get_class_counts(self):
        labels = [segment[1] for segment in self.segments]
        unique_elements, counts = np.unique(labels, return_counts=True)
        occurrences = dict(zip(unique_elements, counts))
        weights = np.zeros(len(id2label))
        for label, count in occurrences.items():
            weights[label2id[label]] = count
        return weights

    def gen_segment(self, idx, format='pth.gz', fps=3.0):
        (start, end), label = self.segments[idx]

        if label == '.': label = 'dot'
        if label == ',': label = 'comma'
        if label == '[s]': label = 'space'
        if label == '[i]': label = 'idle'

        if not os.path.exists(f'{self.data_dir}/segments_{format}/{label}'):
            os.makedirs(f'{self.data_dir}/segments_{format}/{label}')
        
        if format == 'dir':
            for i in range(start, end + 1):
                source = f"{self.videos_dir}/{self.video_name}/frame_{i}.jpg"
                destination = f'{self.data_dir}/segments_{format}/{label}/{self.video_name}_{label}_f{start}_{end}'
                if not os.path.exists(destination): 
                    os.makedirs(destination)
                shutil.copy(source, destination)
        else:
            frames = self._get_frames(start, end)

            if not self.gray_scale:
                frames = frames.permute(0, 2, 3, 1)
            
            video_name = f'{self.data_dir}/segments_{format}/{label}/{self.video_name}_{label}_f{start}_{end}.{format}'
            torchvision.io.video.write_video(filename=video_name, video_array=frames, fps=fps)
        

class KeySegmentDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, ids=None, transforms=None):
        """
            - video_paths: path to videos
            - ids: label id of those videos. If None, the ids will be infered by all `label2id` 
                at the begining of this file. It will include 31 possible classes.
            - transforms: take a tensor of frames (shape = Time, Channel, Height, Width) and apply transforms(frames)
        """
        self.video_paths = video_paths
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_dir = self.video_paths[idx]
        # print('video_dir: ', video_dir)
        
        if self.ids:
            id = self.ids[idx]
        else:
            label = video_dir.split('/')[-2]
            id = label2id[label]

        jpg_files = sorted(glob.glob(os.path.join(video_dir, '*.jpg')))
        frames = []

        for jpg in jpg_files:
            image = torchvision.io.read_image(jpg)
            frames.append(image)

        frames = torch.stack(frames)

        if self.transforms:
            frames = self.transforms(frames)
        return frames, id
    
class BaseKeyDataModule(L.LightningDataModule):
    """
    Base class for key dataset. Subclass must implement setup_data. This function must do the following:
    - Set id2label, label2id
    - Set videos to the paths to all videos
    - Set ids to the id of the label for those videos
    - Call self.split() to perform splitting.
    """

    def __init__(self, segment_dir, transforms=None, batch_size=4, num_workers=4, train_size=0.8):
        """
        Note: test, val set size = (1 - train_size)/2
        """
        super().__init__()
        self.batch_size = batch_size
        self.segment_dir = segment_dir
        self.transforms = transforms
        self.num_workers = num_workers
        self.train_size = train_size
        self.setup_data()

    def setup_data(self): pass
    def split(self):
        """spliting after call setup"""
        train, val_test, train_ids, val_test_ids = train_test_split(
                                self.videos, self.ids,
                                train_size=self.train_size, 
                                random_state=0)
        
        # same size of val and test
        val, test, val_ids, test_ids = train_test_split(
                                val_test, 
                                val_test_ids, 
                                test_size=0.5, 
                                random_state=0)

        train_counts = Counter(train_ids)
        print("Train:\n", train_counts)
        print("Val:\n", Counter(val_ids))
        print("Test:\n", Counter(test_ids))

        train_weights = []
        
        # weight_for_class_i = total_samples / (num_samples_in_class_i * num_classes)
        for id, label in enumerate(self.id2label):
            freq = train_counts[id]
            train_weights.append(len(train_ids) / (freq * len(self.id2label)))

        print('train_weights: \n', train_weights)

        self.train = KeySegmentDataset(train, train_ids, self.transforms)
        self.val = KeySegmentDataset(val, val_ids, self.transforms)
        self.test = KeySegmentDataset(test, test_ids, self.transforms)
        self.train_weights = train_weights
    
    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers else False)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers else False,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True if self.num_workers else False,
                          shuffle=False)
class KeyClfDataModule(BaseKeyDataModule):
    def setup_data(self):
        self.id2label = ['comma', 'dot', 'BackSpace', 'space', 
                         'a', 'b', 'c', 'd', 
                         'e', 'f', 'g', 'h', 
                         'i', 'j', 'k', 'l', 
                         'm', 'n', 'o', 'p', 
                         'q', 'r', 's', 't', 
                         'u', 'v', 'w', 'x', 
                         'y', 'z']
        self.label2id = {
            'comma': 0, 'dot': 1, 'BackSpace': 2, 'space': 3, 
            'a': 4, 'b': 5, 'c': 6, 'd': 7, 
            'e': 8, 'f': 9, 'g': 10, 'h': 11, 
            'i': 12, 'j': 13,  'k': 14, 'l': 15,
            'm': 16, 'n': 17, 'o': 18, 'p': 19, 
            'q': 20, 'r': 21, 's': 22, 't': 23, 
            'u': 24, 'v': 25, 'w': 26, 'x': 27, 
            'y': 28, 'z': 29
        }
        all_videos = sorted(glob.glob(f"{self.segment_dir}/*/*"))
        videos = []
        for video_path in all_videos:
            if not "idle" in video_path:
                videos.append(video_path)
        
        ids = [self.label2id[video.split('/')[-2]] for video in videos]
        self.videos = videos
        self.ids = ids
        self.split()

class KeyDetectDataModule(BaseKeyDataModule):
    def setup_data(self):
        self.id2label = ['idle', 'active']
        self.label2id = {'idle': 0, 'active': 1}
        self.videos = sorted(glob.glob(f"{self.segment_dir}/*/*"))
        self.ids = [0 if "idle" in video_path else 1 for video_path in self.videos]
        self.split()

class KeyClf(L.LightningModule):
    def __init__(self, weights, model_name, id2label, label2id, learning_rate=0.01):
        super().__init__()
        self.num_classes = len(id2label)
        self.label2id = label2id
        self.id2label= id2label
        
        self.model_name = model_name

        self.loss_fn = torch.nn.CrossEntropyLoss(torch.tensor(weights).float())
        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.lr = learning_rate
        
        self.test_preds = []
        self.test_targets = []
        
        self.cur_train_acc = None
        self.cur_val_acc = None

        self.save_hyperparameters()

    def forward(self, batch):
        videos, targets = batch
        preds = self.model(videos)
        pred_ids = torch.argmax(preds, dim=1)
        loss = self.loss_fn(preds, targets.long())
        
        return loss, pred_ids

    def test_step(self, batch):
        _, targets = batch
        loss, pred_ids = self.forward(batch)
        pred_labels = [self.id2label[_id] for _id in pred_ids]
        self.test_preds += pred_labels
        self.test_targets += [self.id2label[_id] for _id in targets]
        
        self.log('test_loss', loss, sync_dist=True, prog_bar=True, on_step=False)

    def on_test_end(self):
        df = pd.DataFrame({"pred": self.test_preds, "target": self.test_targets})
        df.to_csv(f'./{self.model_name}_test_results.csv')
        print(classification_report(self.test_targets, self.test_preds))

    def training_step(self, batch):
        _, targets = batch
        loss, pred_ids = self.forward(batch)
        
        self.cur_train_acc = self.train_acc(pred_ids, targets.long())
        
        self.log('train_loss', loss, 
                 sync_dist=True, prog_bar=True,  on_step=False, on_epoch=True)
        
        self.log('train_acc', self.train_acc, 
                 sync_dist=True, prog_bar=True, on_step=False,  on_epoch=True)
        return loss
    
    def on_train_epoch_end(self) -> None:
        print(f"Epoch {self.current_epoch}, train_acc: {self.cur_train_acc}")

    def validation_step(self, batch):
        _, targets = batch
        loss, pred_ids = self.forward(batch)

        self.cur_val_acc = self.val_acc(pred_ids, targets.long())
        self.log('val_loss', loss,
                 sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, 
                 sync_dist=True, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        print(f"Epoch {self.current_epoch}, val_acc: {self.cur_val_acc}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0.1)


if __name__ == "__main__":
    # all_videos = sorted(glob.glob(f"./datasets/angle/segments_dir/*/*"))
    # labels = [video.split('/')[-2] for video in all_videos]
    # print('all_video: ', all_videos[0])
    # print('labels: ', labels[0])
    
    # for i in range(9):
    #     print('i: ', i)
    #     ds = KeyStreamDataset(f'video_{i}', 
    #                           './datasets/angle', 
    #                           f_before=3, 
    #                           f_after=4,
    #                           gap=2,
    #                           transforms=torchvision.transforms.v2.Compose([
    #                               torchvision.transforms.v2.Resize(224, antialias=True),
    #                             ]),
    #                           crop=[720, 0, 560, 720])
    #     for idx, segment in tqdm(enumerate(ds.segments)):
    #         ds.gen_segment(idx, 'mp4')
    
    
    dm = KeyClfDataModule('datasets/angle/segments_dir', None)
    # dm.setup()
    # frames, id = dm.train[0]
    # print('id: ', dm.id2label[id])
    # torchvision.io.video.write_video('test_clf.mp4', frames.permute(0, 2, 3, 1), 3.0)


    # detect_dm = KeyDetectDataModule('datasets/angle/segments_dir', None)
    # detect_dm.setup()
    # frames, id = detect_dm.train[0]
    # print('id: ', detect_dm.id2label[id])
    # torchvision.io.video.write_video('test_detect.mp4', frames.permute(0, 2, 3, 1), 3.0)
