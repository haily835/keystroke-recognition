from typing import Dict, Any
import torch
import lightning as L
from torch.nn.functional import one_hot
import  torchmetrics
from sklearn.metrics import classification_report
import pandas as pd
from lightning_utils.dataset import *
from models.resnet import *
from pytorchvideo.models import *
import importlib

class KeyClf(L.LightningModule):
    def __init__(self, 
                 name: str, 
                 classpath: str, 
                 init_args: Dict[str, Any], 
                 id2label: str, 
                 label2id: str):
        super().__init__()
        self.name = name

        # Parse classpath and model arguments
        class_module = '.'.join(classpath.split('.')[:-1])
        class_name = classpath.split('.')[-1]
        
        module = importlib.__import__(
            class_module, 
            fromlist=[class_name]
        )
        args_class = getattr(module, class_name)

        model_args = {
            key: eval(str(value)) 
            for key, value in init_args.items()
        }

        self.model = args_class(**model_args)

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

        self.train_losses = []
        self.val_losses = []
        self.save_hyperparameters()

    def forward(self, batch):
        videos, targets = batch
        videos = videos.permute(0, 2, 1, 3, 4)
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
        if not os.path.exists('results'):
            os.mkdir('results')

        df = pd.DataFrame({"pred": self.test_preds, "target": self.test_targets})
        df.to_csv(f'./results/{self.model_name}_test_results.csv')
        print(classification_report(self.test_targets, self.test_preds))

    def training_step(self, batch):
        videos, targets = batch        
        loss, pred_ids = self.forward((videos, targets))
        self.cur_train_acc = self.train_acc(pred_ids, targets.long())
        self.log('train_loss', loss,
                 sync_dist=True, prog_bar=True,  on_step=False, on_epoch=True)

        self.log('train_acc', self.cur_train_acc,
                 sync_dist=True, prog_bar=True, on_step=False,  on_epoch=True)
        
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

    def on_train_epoch_end(self) -> None:
        print(f"EPOCH {self.current_epoch} train_acc {self.cur_train_acc}")

    def on_validation_epoch_end(self) -> None:
        print(f"EPOCH {self.current_epoch} val_acc {self.cur_val_acc}")