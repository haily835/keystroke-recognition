import torch
import lightning as L
from torch.nn.functional import one_hot
import  torchmetrics
from sklearn.metrics import classification_report
import pandas as pd
from lightning_utils.dataset import *
from models.resnet import *

class KeyClf(L.LightningModule):
    def __init__(self, model_name, model_str, id2label, label2id):
        super().__init__()
        self.model_name = model_name

        self.model = eval(model_str)

        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.id2label = eval(id2label)
        self.label2id = eval(label2id)
        self.num_classes = len(id2label)
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes)
        self.val_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes)
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.num_classes)
        

        self.test_preds = []
        self.test_targets = []

        self.cur_train_acc = None
        self.cur_val_acc = None
        self.save_hyperparameters()

    def forward(self, batch):
        videos, targets = batch
        preds = self.model(videos)
        pred_ids = torch.argmax(preds, dim=1)
        loss = self.loss_fn(preds, targets)

        return loss, pred_ids

    def test_step(self, batch):
        videos, targets = batch
        loss, pred_ids = self.forward((videos, targets.long()))
        pred_labels = [self.id2label[_id] for _id in pred_ids]
        self.test_preds += pred_labels
        self.test_targets += [self.id2label[_id] for _id in targets]

        self.log('test_loss', loss, sync_dist=True,
                 prog_bar=True, on_step=False)

    def on_test_end(self):
        df = pd.DataFrame({"pred": self.test_preds, "target": self.test_targets})
        df.to_csv(f'./{self.model_name}_test_results.csv')
        print(classification_report(self.test_targets, self.test_preds))

    def training_step(self, batches):
        all_videos = []
        targets = []
        for b in batches:
            b_videos, b_targets = b
            all_videos.append(b_videos)
            if len(b_targets.size()) == 1:
                b_targets = one_hot(b_targets, num_classes=len(self.id2label))
            targets.append(b_targets)
        all_videos = torch.concat(all_videos)
        
        # print(all_videos.shape)
        
        targets = torch.concat(targets).float()
        
        loss, pred_ids = self.forward((all_videos, targets))
        self.log('train_loss', loss,
                 sync_dist=True, prog_bar=True,  on_step=False, on_epoch=True)

        # self.log('train_acc', self.cur_train_acc,
        #          sync_dist=True, prog_bar=True, on_step=False,  on_epoch=True)
        return loss

    
    def validation_step(self, batch):
        videos, targets = batch
        loss, pred_ids = self.forward((videos, targets.long()))
        self.cur_val_acc = self.val_acc(pred_ids, targets.long())
        self.log('val_loss', loss,
                 sync_dist=True, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True)
        self.log('val_acc', 
                 self.cur_val_acc,
                 sync_dist=True, 
                 prog_bar=True, 
                 on_step=False, 
                 on_epoch=True)
        return loss

    

    