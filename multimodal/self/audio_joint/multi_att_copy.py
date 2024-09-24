import os
import pandas as pd
import numpy as np
from pprint import pprint
from pathlib import Path
from collections import Counter
import pickle
import random
import argparse
import time
from datetime import datetime

# torch:
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer, BertModel, XLMTokenizer, XLMModel, XLMRobertaTokenizer, XLMRobertaModel

# lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

# text
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaTokenizerFast, RobertaTokenizer, RobertaModel
from transformers import XLMTokenizer, XLMModel
from sklearn.model_selection import KFold

# Audio 
#import librosa
#import torchaudio
import transformers 
#from transformers import Wav2Vec2FeatureExtractor, AutoModel

audio_col = ['F0semitoneFrom27.5Hz_sma3nz_amean','F1amplitudeLogRelF0_sma3nz_amean','F1bandwidth_sma3nz_amean','F1frequency_sma3nz_amean',
             'F2amplitudeLogRelF0_sma3nz_amean','F2bandwidth_sma3nz_amean','F2frequency_sma3nz_amean','F3amplitudeLogRelF0_sma3nz_amean',
             'F3bandwidth_sma3nz_amean','F3frequency_sma3nz_amean','HNRdBACF_sma3nz_amean','alphaRatioV_sma3nz_amean',
             'hammarbergIndexV_sma3nz_amean','jitterLocal_sma3nz_amean','logRelF0-H1-A3_sma3nz_amean','logRelF0-H1-H2_sma3nz_amean',
             'loudness_sma3_amean','mfcc1_sma3_amean','mfcc2_sma3_amean','mfcc3_sma3_amean','mfcc4_sma3_amean','shimmerLocaldB_sma3nz_amean',
             'slopeV0-500_sma3nz_amean','slopeV500-1500_sma3nz_amean','spectralFlux_sma3_amean'] #, 'duration', 'pause']

# default arguments that will get changed
class Arg:
    version = 1
    # data
    epochs: int = 15  # Max Epochs, BERT paper setting [3,4,5]
    max_length: int = 500  # Max Length input size
    report_cycle: int = 30  # Report (Train Metrics) Cycle
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    hidden_size = 256 # BERT-base: 768, BERT-large: 1024, BERT paper setting
    batch_size: int = 16
    t_hidden_size = 768
    t_x_hidden_size = 1280
    
    
    

    # 300/16 > 200/32 > 512/8
class TextModel(LightningModule):
    def __init__(self, args,config):
        super().__init__()
        # config:
        
        self.args = args
        self.config = config
        self.batch_size = self.args.batch_size
        
        # meta data:
        self.epochs_index = 0
        self.hidden_size = self.args.hidden_size
        print(self.embed_type)
        
        # dataset
        self.data_path = self.config['data_path']
        self.label_col = self.config['label_col'] # 'current_bp_emo_y' 
        self.label_names = ['TD','LLE'] # ['bp_remission','bp_manic','bp_anxiety','bp_irritability','bp_depressed']
        self.num_labels = self.config['num_labels']
        self.col_name= self.config['text_col']
        
        # model 
        if self.embed_type == "bert":
            pretrained = "bert-base-uncased"
            self.t_tokenizer = BertTokenizer.from_pretrained(pretrained)
            self.model = BertModel.from_pretrained(pretrained)

        elif self.embed_type == "roberta":
            pretrained = 'roberta-base'
            self.t_tokenizer = RobertaTokenizer.from_pretrained(pretrained)
            self.model = RobertaModel.from_preigned(pretrained)
            
        elif self.embed_type == "mbert":
            pretrained = 'bert-base-multilingual-uncased'
            self.t_tokenizer = BertTokenizer.from_pretrained(pretrained)
            self.model = BertModel.from_pretrained(pretrained)
            
        elif self.embed_type == "xlm":
            pretrained = 'xlm-mlm-100-1280'
            self.t_tokenizer = XLMTokenizer.from_pretrained(pretrained)
            self.model = XLMModel.from_pretrained(pretrained)
        
        elif self.embed_type == "xlmr":
            pretrained = 'xlm-roberta-base'
            self.t_tokenizer = XLMRobertaTokenizer.from_pretrained(pretrained)
            self.model = XLMRobertaModel.from_pretrained(pretrained)
        
    def forward(self, input_ids, mask=None, **kwargs): 

        output= self.model(input_ids, attention_mask = mask)
        last_hidden_states=output.last_hidden_state
        
        return last_hidden_states


class FusionModel(LightningModule):
    def __init__(self, args, config):
        super(FusionModel, self).__init__()
        
        self.config=config
        self.args=args
        
        # Data
        self.data_path=self.config['data_path']
        self.label_col = self.config['label_col']
        self.num_labels = self.config['num_labels']
        self.label_names = ['TD','LLE']
        self.audio_feats = self.config['audio_feats']
        self.joint_feats = self.config['joint_feats']
        self.text_feats = self.config['text_feats']
        
        self.joint_nh = self.config['joint_nh']
        self.audio_nh = self.config['audio_nh']
        self.text_nh = self.config['text_nh']
        
        self.joint_dp = self.config['joint_dp']
        self.audio_dp = self.config['audio_dp']
        self.text_dp = self.config['text_dp']

        
        # meta data
        self.batch_size=self.args.batch_size
        self.hidden_size = self.args.hidden_size

    
        # Models 
        #self.text_model = TextModel(args,config)
        
        self.pool=nn.AdaptiveMaxPool2d((1, self.hidden_size)) #ex) [16, 500, 768] -> [16, 1, self.hidden_size] 
        
        self.mha_j = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.joint_nh, dropout=self.joint_dp) 
        self.mha_a = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.audio_nh, dropout=self.audio_dp)
        self.mha_t = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.text_nh, dropout=self.text_dp)

        self.dropout = nn.Dropout(self.joint_dp)
        self.j_dense0 = nn.Linear(self.joint_feats, self.hidden_size)
        self.a_dense0 = nn.Linear(self.audio_feats, self.hidden_size)          
        self.t_dense0 = nn.Linear(self.text_feats, self.hidden_size) 

        self.dense1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, self.num_labels)
        
        # Add & Norm layers
        self.norm_t = nn.LayerNorm(self.hidden_size)
        self.norm_a = nn.LayerNorm(self.hidden_size)

    def forward(self, joint_input_ids, audio_input_ids, text_input_ids, mask=None, **kwargs):
        
        # Text Embeddings
        #text_embed = self.text_model(text_input_ids, mask=mask) 
        #text_embed = self.pool(text_embed)
        
        joint_embed = self.dropout(self.j_dense0(joint_input_ids))
        joint_embed_att, _ = self.mha_j(joint_embed, joint_embed, joint_embed)
        joint_embed = self.norm_t(joint_embed + joint_embed_att)  # Add & Norm

        # Audio Embeddings
        audio_embed = self.dropout(self.a_dense0(audio_input_ids))
        audio_embed_att, _ = self.mha_a(audio_embed, audio_embed, audio_embed) 
        audio_embed = self.norm_a(audio_embed + audio_embed_att)  # Add & Norm


        #text_embed = self.dropout(self.t_dense0(text_input_ids))
        #text_embed_att, _ = self.mha_t(text_embed, text_embed, text_embed) 
        #text_embed = self.norm_a(text_embed + text_embed_att)  # Add & Norm
  
        
        fuse1 = torch.stack((joint_embed, audio_embed), dim=1) 
        fuse1_mean, fuse1_std = torch.std_mean(fuse1, dim=1)
        fuse = torch.cat((fuse1_mean, fuse1_std), dim=1) 
        
        logits=self.dense3(self.dropout(self.dense2(self.dropout(self.dense1(fuse)))))
        
        return logits

        
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=self.config['gamma'])

        return [optimizer], [{'scheduler': scheduler, 'interval': "epoch"}]
    


    def preprocess_dataframe(self):

        # set up dataframe
        df = self.args.data_updated 
        df_train = df[df['ex'] == 'train']
        df_test = df[df['ex'] == 'test']

        print(f'# of train:{len(df_train)}, test:{len(df_test)}')

        # get features and set them to train and test
        df['full_text_feature_scaled'] = df['full_text_feature_scaled'].values.tolist()
        	
        #df['text_feature_scaled'] = df['text_feature_scaled'].values.tolist()
        df['joint_feature_scaled'] = df['joint_feature_scaled'].values.tolist()
        df['audio_feature_scaled'] = df['audio_feature_scaled'].values.tolist()


        self.train_data = TensorDataset(
            torch.tensor(np.array(df_train['joint_feature_scaled'].tolist()), dtype=torch.float),
            torch.tensor(np.array(df_train['audio_feature_scaled'].tolist()), dtype=torch.float),
            torch.tensor(np.array(df_train['full_text_feature_scaled'].tolist()), dtype=torch.float),
            torch.tensor(np.array(df_train[self.label_col].tolist()), dtype=torch.long),
        )

        self.test_data = TensorDataset(
             torch.tensor(np.array(df_test["joint_feature_scaled"].tolist()), dtype=torch.float),
             torch.tensor(np.array(df_test['audio_feature_scaled'].tolist()), dtype=torch.float),
             torch.tensor(np.array(df_test['full_text_feature_scaled'].tolist()), dtype=torch.float),
             torch.tensor(np.array(df_test[self.label_col].tolist()), dtype=torch.long)
        )

        
    def train_dataloader(self):
        
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.cpu_workers,
        )
    
    def test_dataloader(self):

        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.cpu_workers,
        )
    
    def training_step(self, batch, batch_idx):
        joint_token, audio_token, text_token, labels = batch  
        logits = self(joint_input_ids=joint_token, audio_input_ids=audio_token, text_input_ids=text_token) 
        loss = nn.CrossEntropyLoss()(logits, labels)   

        self.log("train_loss", loss)
        return {'loss': loss}

            

    def test_step(self, batch, batch_idx):
        joint_token, audio_token, text_token, labels = batch
        logits = self(joint_input_ids=joint_token, audio_input_ids=audio_token, text_input_ids=text_token) 
        preds = logits.argmax(dim=-1)

        y_pred = list(preds.cpu().numpy())
        labels= list(labels.cpu().numpy())
        
        return {
            'y_true': labels,
            'y_pred': y_pred
        }


    def test_epoch_end(self, outputs):

        # get all predictions 
        y_pred = []
        y_true =[]

        for i in outputs:
            y_pred += i['y_pred']
            y_true += i['y_true']
        
        y_pred = np.asanyarray(y_pred)
        y_true = np.asanyarray(y_true)
        
        # predictions df for each person
        df = self.args.data_updated 
        persons = np.array(df[df['ex'] == 'test']['name'].tolist())
        pred_dict = {}
        pred_dict['y_pred']= y_pred
        pred_dict['y_true']= y_true
        pred_dict['name'] = persons
        pred_df = pd.DataFrame(pred_dict)

        # print results 
        print(classification_report(y_true,y_pred))
        metrics_dict = classification_report(y_true, y_pred,zero_division=1,
                                             target_names = self.label_names, 
                                             output_dict=True)
        df_result = pd.DataFrame(metrics_dict).transpose()
        df_result = df_result.applymap(lambda x: f" {x} " if isinstance(x, (int, float)) else x)
    
        df_result = df_result.reset_index()
        df_result.columns = ['data','precision', 'recall', 'f1-score', 'support']
    
        # save results only if savePsplits
        if self.config['save_splits']: 

            pred_df.to_csv(f'att_{self.args.index}_{self.args.random_seed}_{self.args.heads}_{self.args.epochs}_{self.args.hidden_size}_{self.args.batch_size}_{self.args.lr}_{self.args.gamma}test_pred.csv', index=False)

            df_result.to_csv(f'att_{self.args.index}_{self.args.random_seed}_{self.args.heads}_{self.args.epochs}_{self.args.hidden_size}_{self.args.batch_size}_{self.args.lr}_{self.args.gamma}test.csv')
 
        # test result will be appended in main along with all splits
        self.test_results = df_result

        # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        # specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        # self.specificity = specificity



        # cm = confusion_matrix(y_true, y_pred)
        # tn = cm.diagonal()
        # fp = cm.sum(axis=0) - tn
        # fn = cm.sum(axis=1) - tn
        # tp = cm.sum() - (fp + fn + tn)

        # # Calculate specificity for each class
        # specificity = tn / (tn + fp)  # Handling divide by zero is needed

        # # Calculate weighted average of specificity
        # support = cm.sum(axis=1)
        # weighted_specificity = np.average(specificity, weights=support)

        # # Store the result
        # self.specificity = weighted_specificity

        y_true_ = np.array(y_true)  # Ensure y_test is a numpy array for proper indexing
        y_pred_ = np.array(y_pred)  # Ensure y_pred is a numpy array for consistency
        cm = confusion_matrix(y_true_, y_pred_)
    
        specificity_per_class = []
        for i in range(len(cm)):
            TN = cm.sum() - (cm[i,:].sum() + cm[:,i].sum() - cm[i,i])
            FP = cm[:,i].sum() - cm[i,i]
            specificity_per_class.append(TN / (TN + FP))
            
        # Weighted Specificity
        weights = np.array([np.sum(y_true_ == k) for k in range(cm.shape[0])])

        weighted_specificity = np.sum(specificity_per_class * weights) / np.sum(weights)
        self.specificity = weighted_specificity


def main(args,config):
    
    print()
    print("MODEL STATS\n=======================================")
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    print()
    seed_everything(config['random_seed'])
    
    # modify arguments based off config 
    args.data_path = config['data_path']
    args.random_seed = config['random_seed']
    args.epochs = config['epochs']
    args.lr = config['lr']
    args.hidden_size = config['hidden_size']
    args.batch_size = config['batch_size']
    args.heads = config['heads']
    args.gamma = config['gamma']


    #config['random_seed']
    # get the data and set the folds 
    df_data = pd.read_json(config['data_path'])
    group_kfold = StratifiedGroupKFold(n_splits=config['splits'], shuffle=True, random_state=config['random_seed'])
    groups = df_data['name'].tolist()

    # list that will hold results of all splits
    dt_results = []
    
    # i indicates split
    spec = 0
    i = 0

    # iterate through each group kfold: label used for balancing split and group used for separation of groups 
    for train_idxs, test_idxs in group_kfold.split(df_data['audio_feature_scaled'], df_data['label'], groups):

        # set up trainer and early stopping 
        early_stop_callback = EarlyStopping(
            monitor='train_loss',
            min_delta=0.00,
            patience=35,
            verbose=True,
            mode='min'
        )
        trainer = Trainer(
            logger=False,
            enable_checkpointing=False,  # Disable checkpointing as there's no validation to monitor
            max_epochs=args.epochs,
            fast_dev_run=args.test_mode,
            callbacks=[early_stop_callback],
            deterministic=False,
            gpus=[config['gpu']] if torch.cuda.is_available() else None,
            precision=16 if args.fp16 else 32,
            limit_val_batches=0,  # This effectively disables validation
        )

        # (re)set train and test category and verify balance 
        train_td = 0
        train_lle = 0

        test_td = 0
        test_lle = 0 
        for index, data in df_data.iterrows():
            if index in train_idxs:
                df_data.at[index, 'ex'] = "train"
                if df_data.at[index, 'label'] == 1:
                    train_td += 1
                elif df_data.at[index, 'label'] == 0:
                    train_lle += 1
            elif index in test_idxs:
                df_data.at[index, 'ex'] = "test"
                if df_data.at[index, 'label'] == 1:
                    test_td += 1
                elif df_data.at[index, 'label'] == 0:
                    test_lle += 1

        print(train_td, train_lle)
        print(test_td, test_lle)
        #name = f"data_{i}.csv"
        #df_data['ex'].to_csv(name, index=False)

        # provide the index iteration and reset dataframe for preprocessing
        args.index = i
        args.data_updated = df_data

        # (re)set up model
        model=FusionModel(args, config)
        
        # (re)process dataframe 
        print(":: Processing Data ::")
        model.preprocess_dataframe()

        # start training model
        print(":: Start Training ::")
        trainer.fit(model)
        result = trainer.test(model, dataloaders=model.test_dataloader())

        dt_results.append(model.test_results)
        spec += model.specificity
        i+=1

    # calculate total specificity
    total_spec = spec/config['splits']

    # create meaned dataframe of all the splits along with specificity
    combined_df = pd.concat(dt_results, ignore_index=True)
    numeric_columns = ['precision', 'recall', 'f1-score', 'support']

    for column in numeric_columns:
        combined_df[column] = pd.to_numeric(combined_df[column], errors='coerce')

    average_df = combined_df.groupby('data', as_index=False)[numeric_columns].mean()
    average_df['specificity'] = total_spec


    name = f"results_{config['random_seed']}_{config['heads']}_{config['epochs']}_{config['hidden_size']}_{config['batch_size']}_{config['lr']}_{config['gamma']}.csv"

    average_df.to_csv(name, index=False)
    print(spec/config['splits'])
    
if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # settings 
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=2023) 
    parser.add_argument("--gamma", type=float, default=.99, help="gamma for optimizer")

    # models 
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="epoch")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--joint_feats", type=int, default=30)
    parser.add_argument("--audio_feats", type=int, default=25)
    parser.add_argument("--text_feats", type=int, default=345)

    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--joint_nh", type=int, default=8)
    parser.add_argument("--audio_nh", type=int, default=8)
    parser.add_argument("--text_nh", type=int, default=8)

    parser.add_argument("--joint_dp", type=float, default=0.1)
    parser.add_argument("--audio_dp", type=float, default=0.1)
    parser.add_argument("--text_dp", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    
    
    # datasets 
    parser.add_argument("--data_path", type=str, default="/home/benjaminbarrera-altuna/Desktop/LLE/dataset/data_50.json")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--save_splits", type=bool, default=False) 
    parser.add_argument("--splits", type=int, default=5) 
    
    config = parser.parse_args()
    args = Arg()
    
    main(args,config.__dict__)     