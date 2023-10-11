import torch
import pandas as pd
import os
from train_model_lightning import TrainModelLigthning, CustomTimeCallback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt
import lightning as L

from train_model_lightning import TrainModelLigthning

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  

device = torch.device(dev)
print("Device: {}".format(device))

#Class for training and test 
class PytorchTrainingAndTest:
    
    def run_model(self, exp_num, model, model_name, database_name, train, test, learning_rate, num_epochs, num_class=2):
        '''
          function to train the model using pytorch lightning
          params:
            - exp_num: num of experiments to run a model
            - model: architecture pre-trained that trained on dataset
            - database_name: string that describe the databaset to train a model
            - train: dataloader of the train
            - test: dataloader of the test
            - learning_rate: float to define a learning rate of the model optimization
            - num_epochs: number maximium of epoch during the trainning
            - num_class: class number on dataset
        '''
        
        #init model using a pytorch lightining call
        ligh_model = TrainModelLigthning(model_pretrained=model, 
                                         num_class=num_class, 
                                         lr=learning_rate)
        
        #define callback for earlystopping
        early_stop_callback = EarlyStopping(monitor='val_acc', min_delta=0.01, patience=5, verbose=True, mode='max')
        
        #define custom callback to calculate the train and test time 
        timer = CustomTimeCallback("./metrics/time/train_time_{}-{}.csv".format(model_name, database_name),
                                       "./metrics/time/test_time_{}-{}.csv".format(model_name, database_name))
            
        #Define callback to save the best model weights
        ckp = ModelCheckpoint(dirpath="trained-weights", 
                                  filename="{}-{}-exp{}".format(model_name, database_name, exp_num), 
                                  save_top_k=1, 
                                  mode="max", 
                                  monitor="val_acc",
                                )
            
        #initate callbacks to execute the training
        callbacks=[early_stop_callback, ckp, timer]
        
        #define the function to save the logs
        logger = CSVLogger(save_dir="./metrics/logs/", name="{}-{}".format(model_name, database_name), version=exp_num)
        
        trainer = L.Trainer(
            max_epochs= num_epochs,
            accelerator="gpu",
            devices="auto",
            min_epochs=5,
            log_every_n_steps=10,
            logger=logger,
            deterministic=False,
            callbacks=callbacks
        )
            
        trainer.fit(
            model=ligh_model,
            train_dataloaders=train,
            val_dataloaders=test
        )
        
        metrics = trainer.logged_metrics
        print(metrics)
            
        results =  {
                "exp_num": exp_num,
                "model_name" : model_name,
                "train_acc" : metrics["acc"].item(),
                "train_f1-score": metrics["f1_score"].item(),
                "train_loss":  metrics["loss"].item(),
                "train_precision":  metrics["precision"].item(),
                "train_recall" :  metrics["recall"].item(),
                "train_auc":  metrics["auc"].item(),
                "train_spc":  metrics["specificity"].item(),
                "val_acc" :  metrics["val_acc"].item(),
                "val_f1-score":  metrics["val_f1_score"].item(),
                "val_loss":  metrics["val_loss"].item(),
                "val_precision":  metrics["val_precision"].item(),
                "val_recall" :  metrics["val_recall"].item(),
                "val_auc":  metrics["val_auc"].item(),
                "val_spc":  metrics["val_specificity"].item(),
        }
        results = {k:[v] for k,v in results.items()}
        metrics_df = pd.DataFrame(results)
        
        metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")

        self.save_metrics_to_figure(metrics, model_name, database_name)
        
        return metrics_df
      
    def save_metrics_to_figure(self, metrics, model_name, database_name):
        '''
          Generate and save figures of metrics accuracy, loss, and f1-score by models and database
          params:
            - metris: dict of metrics extract from pytoch
            - log_dir: path to save images
            - model_name: trained model name
            - database_name: database applied to train the model
        '''
        aggreg_metrics = []
        agg_col = "epoch"
        for i, dfg in metrics.groupby(agg_col):
            agg = dict(dfg.mean())
            agg[agg_col] = i
            aggreg_metrics.append(agg)

        #save 
        df_metrics = pd.DataFrame(aggreg_metrics)
        fig_loss = df_metrics[["loss", "val_loss"]].plot(grid=True, legend=True, xlabel='Epoch', ylabel='Loss').get_figure()
        plt.tight_layout()
        fig_loss.savefig(os.path.join("./", "metrics", "figures", "{}-{}-{}.png".format('loss_metrics',model_name, database_name)))
        
        fig_acc = df_metrics[["acc", "val_acc"]].plot(grid=True, legend=True, xlabel='Epoch', ylabel='Accuracy').get_figure()
        plt.tight_layout()
        fig_acc.savefig(os.path.join("./", "metrics", "figures", "{}-{}-{}.png".format('acc_metrics',model_name, database_name)))
        
        fig_acc = df_metrics[["f1_score", "val_f1_score"]].plot(grid=True, legend=True, xlabel='Epoch', ylabel='F1-Score').get_figure()
        plt.tight_layout()
        fig_acc.savefig(os.path.join("./", "metrics", "figures", "{}-{}-{}.png".format('f1_metrics',model_name, database_name)))
        
