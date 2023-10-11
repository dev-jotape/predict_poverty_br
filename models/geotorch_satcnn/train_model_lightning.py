from typing import Any, Optional
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import time
import os
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import lightning as L
from torch.nn import functional as F
from torchmetrics import Accuracy, Recall, Specificity, Precision, F1Score, AUROC, MatthewsCorrCoef
from lightning.pytorch.callbacks import Callback
# from medmnist import Evaluator
torch.manual_seed(123)

class TrainModelLigthning(L.LightningModule):
    def __init__(self, model_pretrained, num_class, lr):
        super().__init__()
        self.model = model_pretrained
        self.lr = lr
        self.num_class = num_class
        self.criterion = torch.nn.CrossEntropyLoss() if self.num_class > 2 else torch.nn.BCEWithLogitsLoss()
        
        self.train_accuracy = Accuracy(task="binary") if not num_class > 2 else Accuracy(task="multiclass", num_classes=num_class)
        self.val_accuracy = Accuracy(task="binary") if not num_class > 2 else Accuracy(task="multiclass", num_classes=num_class)
        self.train_recall = Recall(task="binary")  if not num_class > 2 else Recall(task="multiclass", num_classes=num_class, average='macro')
        self.val_recall =  Recall(task="binary")  if not num_class > 2 else Recall(task="multiclass", num_classes=num_class, average='macro')
        self.train_specificity = Specificity(task="binary") if not num_class > 2 else Specificity(task="multiclass", num_classes=num_class, average='macro')
        self.val_specificity = Specificity(task="binary") if not num_class > 2 else Specificity(task="multiclass", num_classes=num_class, average='macro')
        self.train_precision = Precision(task="binary") if not num_class > 2 else Precision(task="multiclass", num_classes=num_class, average="macro")
        self.val_precision = Precision(task="binary") if not num_class > 2 else Precision(task="multiclass", num_classes=num_class, average="macro")
        self.train_f1 = F1Score(task="binary") if not num_class > 2 else F1Score(task="multiclass", num_classes=num_class, average="macro")
        self.val_f1 = F1Score(task="binary") if not num_class > 2 else F1Score(task="multiclass", num_classes=num_class, average="macro")
        self.train_auc = AUROC(task="binary") if not num_class> 2 else AUROC(task="multiclass", num_classes=num_class, average="macro")
        self.val_auc = AUROC(task="binary") if not num_class> 2 else AUROC(task="multiclass", num_classes=num_class, average="macro")
        #self.trian_mcc = MatthewsCorrCoef(task="binary") if not num_class > 2 else MatthewsCorrCoef(task="multiclass", num_classes=num_class)
        #self.val_mcc = MatthewsCorrCoef(task="binary") if not num_class > 2 else MatthewsCorrCoef(task="multiclass", num_classes=num_class)

    def forward(self, x):
        return self.model(x)
    
    def _shared_step(self, batch):
        features, y_true = batch
        y_true = y_true if self.num_class > 2 else y_true.view(-1, 1).float()
        logits = self(features)
        loss = self.criterion(logits, y_true)
        y_pred = torch.argmax(logits, dim=1) if self.num_class > 2 else torch.argmax(logits, dim=1).view(-1, 1).float()
        probs = torch.softmax(logits, dim=1) if self.num_class > 2 else torch.sigmoid(logits)
        
        return loss, y_true, y_pred, probs

    def training_step(self, batch, batch_idx):
        loss, y_true, y_pred, probs = self._shared_step(batch)

        self.model.eval()
        with torch.no_grad():
             _, y_true, y_pred, probs = self._shared_step(batch)
        
        #self.train_loss.append(loss_value.item())
        self.train_accuracy.update(y_pred, y_true)
        self.train_precision.update(y_pred, y_true)
        self.train_recall.update(y_pred, y_true)
        self.train_specificity.update(y_pred, y_true)
        self.train_f1.update(y_pred, y_true)
        self.train_auc.update(probs, y_true)
        
        
        self.log('loss', loss, prog_bar=True)
        self.log('acc', self.train_accuracy.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('precision', self.train_precision.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('recall', self.train_recall.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('f1_score', self.train_f1.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('specificity', self.train_specificity.compute(), prog_bar=True, on_epoch=True, on_step=False)
        self.log('auc', self.train_auc.compute(), prog_bar=True, on_epoch=True, on_step=False)
        
        self.model.train()
        
        return loss
   
    def validation_step(self, batch, batch_idx):
        
        loss, y_true, y_pred, probs = self._shared_step(batch)

        self.val_accuracy(y_pred, y_true)
        self.val_precision(y_pred, y_true)
        self.val_recall(y_pred, y_true)
        self.val_specificity(y_pred, y_true)
        self.val_f1(y_pred, y_true)
        self.val_auc(probs, y_true)
        
        self.log('val_loss', loss)
        self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_precision', self.val_precision, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_recall', self.val_recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_f1_score', self.val_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_specificity', self.val_specificity, on_epoch=True, on_step=False, prog_bar=True)
        self.log('val_auc', self.val_auc, on_epoch=True, on_step=False, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_id):
        loss, y_true, y_pred, probs = self._shared_step(batch)
        
        self.val_accuracy(y_pred, y_true)
        self.val_precision(y_pred, y_true)
        self.val_recall(y_pred, y_true)
        self.val_specificity(y_pred, y_true)
        self.val_f1(y_pred, y_true)
        self.val_auc(probs, y_true)
        
        self.log('test_acc', self.val_accuracy, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_precision', self.val_precision, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_recall', self.val_recall, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_f1_score', self.val_f1, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_specificity', self.val_specificity, on_epoch=True, on_step=False, prog_bar=True)
        self.log('test_auc', self.val_auc, on_epoch=True, on_step=False, prog_bar=True)
        
        
        return loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        #optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        #optimizer = optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        miletones = [0.5 * 100, 0.75 * 100]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=miletones, gamma=0.1)
        
        return [optimizer], [scheduler]


class CustomTimeCallback(Callback):
    
    def __init__(self, file_train, file_test) -> None:
        super().__init__()
        if not os.path.exists("./metrics/time"):
            os.mkdir("./metrics/time")
        
        self.file_train = file_train
        self.file_test = file_test
    
    def on_train_start(self, trainer, pl_module):
        self.start_train = time.time()
    
    def on_train_end(self, trainer, pl_module):
        self.train_end = time.time()
        total = (self.train_end - self.start_train)/60
        
        with open(self.file_train, "a") as f:
            f.write("{}\n".format(total))
    
    def on_validation_start(self, trainer, pl_module):
        self.start_test = time.time()
    
    def on_validation_end(self, trainer, pl_module):
        self.test_end = time.time()
        total = (self.test_end - self.start_test)/60
        
        with open(self.file_test, "a") as f:
            f.write("{}\n".format(total))