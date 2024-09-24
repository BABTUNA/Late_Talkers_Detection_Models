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

# text
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaTokenizerFast, RobertaModel
from transformers import XLMTokenizer, XLMModel

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
        
        self.joint_nh = self.config['joint_nh']
        self.audio_nh = self.config['audio_nh']
        
        self.joint_dp = self.config['joint_dp']
        self.audio_dp = self.config['audio_dp']
        
        
        # meta data
        self.batch_size=self.args.batch_size
        self.hidden_size = self.args.hidden_size

    
        # Models 
        #self.text_model = TextModel(args,config)
        
        self.pool=nn.AdaptiveMaxPool2d((1, self.hidden_size)) #ex) [16, 500, 768] -> [16, 1, self.hidden_size] 
        
        self.mha_t = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.joint_nh, dropout=self.joint_dp) 
        self.mha_a = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.audio_nh, dropout=self.audio_dp)
        self.mha_at = nn.MultiheadAttention(embed_dim=self.hidden_size * 2, num_heads=self.audio_nh, dropout=self.audio_dp)


        self.a_dense0 = nn.Linear(self.audio_feats, self.hidden_size)
        self.t_dense0 = nn.Linear(self.joint_feats, self.hidden_size)          

        self.dense1 = nn.Linear(self.hidden_size*2, self.hidden_size)
        self.dense2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.dense3 = nn.Linear(self.hidden_size, self.num_labels)
        
        # Add & Norm layers
        self.norm_t = nn.LayerNorm(self.hidden_size)
        self.norm_a = nn.LayerNorm(self.hidden_size)

    def forward(self, joint_input_ids, audio_input_ids, mask=None, **kwargs):
        
        # Text Embeddings
        #text_embed = self.text_model(text_input_ids, mask=mask) 
        #text_embed = self.pool(text_embed)
        
        joint_embed = self.t_dense0(joint_input_ids)
        audio_embed = self.a_dense0(audio_input_ids)
        
        joint_embed, _ = self.mha_t(joint_embed, audio_embed, audio_embed)
        joint_embed = self.norm_t(joint_embed + joint_embed)  # Add & Norm
        joint_embed2 = torch.mean(joint_embed, dim=1)

        # Audio Embedding
        audio_embed, _ = self.mha_a(audio_embed, joint_embed, joint_embed) 
        audio_embed = self.norm_a(audio_embed + audio_embed)  # Add & Norm
        audio_embed2 = audio_embed    


        #print("audio: ",audio_embed.shape)
        #print(audio_embed2.size(), text_embed2.size())
        
        fuse1 = torch.stack((joint_embed, audio_embed2), dim=1) 
        fuse1_mean, fuse1_std = torch.std_mean(fuse1, dim=1)
        fuse = torch.cat((fuse1_mean, fuse1_std), dim=1) 
        cross, _ = self.mha_at(fuse, fuse, fuse)
        logits=self.dense3(self.dense2(self.dense1(cross)))
        
        return logits
    
    
    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.config['lr'])
        scheduler = ExponentialLR(optimizer, gamma=0.5)
        
        return {
            'optimizer': optimizer,
            'scheduler': scheduler,
        }

    def preprocess_dataframe(self):
        
        df = pd.read_json(self.data_path)
        
        
        df_train = df[df['ex'] == 'train']
        df_test = df[df['ex'] == 'test']

        print(f'# of train:{len(df_train)}, test:{len(df_test)}')

        df['joint_feature_scaled'] = df['joint_feature_scaled'].values.tolist()
        df['audio_feature_scaled'] = df['audio_feature_scaled'].values.tolist()

        self.train_data = TensorDataset(
            torch.tensor(np.array(df_train['joint_feature_scaled'].tolist()), dtype=torch.float),
            torch.tensor(np.array(df_train['audio_feature_scaled'].tolist()), dtype=torch.float),
            torch.tensor(np.array(df_train[self.label_col].tolist()), dtype=torch.long),
        )

        self.test_data = TensorDataset(
             torch.tensor(np.array(df_test["joint_feature_scaled"].tolist()), dtype=torch.float),
             torch.tensor(np.array(df_test['audio_feature_scaled'].tolist()), dtype=torch.float),
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
        joint_token, audio_token,  labels = batch  
        logits = self(joint_input_ids=joint_token, audio_input_ids=audio_token) 
        loss = nn.CrossEntropyLoss()(logits, labels)   

        self.log("train_loss", loss)
        return {'loss': loss}

            

    def test_step(self, batch, batch_idx):
        joint_token, audio_token, labels = batch
        logits = self(joint_input_ids=joint_token, audio_input_ids=audio_token) 
        preds = logits.argmax(dim=-1)

        y_pred = list(preds.cpu().numpy())
        labels= list(labels.cpu().numpy())
        
        return {
            'y_true': labels,
            'y_pred': y_pred
        }



    def test_epoch_end(self, outputs):
        y_pred = []
        y_true =[]

        for i in outputs:
            y_pred += i['y_pred']
            y_true += i['y_true']
        
        y_pred = np.asanyarray(y_pred)
        y_true = np.asanyarray(y_true)
        
        df = pd.read_json(self.data_path)
        persons = np.array(df[df['ex'] == 'test']['name'].tolist())
        pred_dict = {}
        pred_dict['y_pred']= y_pred
        pred_dict['y_true']= y_true
        pred_dict['name'] = persons
        pred_df = pd.DataFrame(pred_dict)
        pred_df.to_csv(
            f'cros_{self.args.random_seed}_{self.args.heads}_{self.args.epochs}_{self.args.hidden_size}_{self.args.batch_size}_test_pred.csv', index=False)

        print(classification_report(y_true,y_pred))
        metrics_dict = classification_report(y_true, y_pred,zero_division=1,
                                             target_names = self.label_names, 
                                             output_dict=True)
        df_result = pd.DataFrame(metrics_dict).transpose()
        df_result = df_result.applymap(lambda x: f" {x} " if isinstance(x, (int, float)) else x)
        df_result.to_csv(
            f'cros_{self.args.random_seed}_{self.args.heads}_{self.args.epochs}_{self.args.hidden_size}_{self.args.batch_size}_test.csv')
 


    
def main(args,config):
    
    print()
    print("MODEL STATS\n=======================================")
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", config['random_seed'])
    print()
    seed_everything( config['random_seed'])
    
    args.data_path = config['data_path']
    
    args.random_seed = config['random_seed']
    args.epochs = config['epochs']
    args.hidden_size = config['hidden_size']
    args.batch_size = config['batch_size']
    args.heads = config['heads']
    
    model=FusionModel(args, config)
    
    print(":: Processing Data ::")
    model.preprocess_dataframe()


    print(":: Start Training ::")

    early_stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.00,
        patience=10,
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
    trainer.fit(model)
    trainer.test(model, dataloaders=model.test_dataloader())
    
if __name__ == '__main__': 

    parser = argparse.ArgumentParser("main.py", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # settings 
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--random_seed", type=int, default=2023) 
    
    # models 
    parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="epoch")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--joint_feats", type=int, default=30)
    parser.add_argument("--audio_feats", type=int, default=25)

    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--joint_nh", type=int, default=8)
    parser.add_argument("--audio_nh", type=int, default=8)
    parser.add_argument("--joint_dp", type=float, default=0.1)
    parser.add_argument("--audio_dp", type=float, default=0.1)
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
    
    
    # datasets 
    parser.add_argument("--data_path", type=str, default="/home/benjaminbarrera-altuna/Desktop/LLE/dataset/full_data.json")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--num_labels", type=int, default=2)
    
    config = parser.parse_args()
    print(config)
    args = Arg()
    
    main(args,config.__dict__)       