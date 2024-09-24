from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
#from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# save minimal results to dataframe
def save_results_(model, val_score, config):
    print(val_score)


    accuracy = val_score.pop('accuracy')
    
    val_df = pd.DataFrame.from_dict(val_score,dtype=object).T.reset_index()
    val_df = val_df.rename(columns = {'index':'category'})
    val_df['accuracy'] = accuracy
    val_df['model'] = model
    val_df['token_size'] = config['bind']

    save_time = datetime.now().__format__("%m%d_%H%M%S%Z")
    pd.DataFrame(val_df).to_csv(f'_{model}_.csv',index=False)  

# save holistic results to data frame
def save_results(model, y_names, val_score, y_train, y_test,y_pred, config):
    print(val_score)


    accuracy = val_score.pop('accuracy')
    
    val_df = pd.DataFrame.from_dict(val_score,dtype=object).T.reset_index()
    val_df = val_df.rename(columns = {'index':'category'})
    val_df['accuracy'] = accuracy
    val_df['model'] = model
    val_df['token_size'] = config['bind']
    val_df['train_size'] = len(y_train)
    val_df['test_size'] = len(y_pred)
    val_df['train_%'] = (len(y_train)) / (len(y_train) + len(y_pred))
    val_df['test_%'] = (len(y_pred)) / (len(y_train) + len(y_pred))


    pred_dict = { 
        'model' : [model]*len(y_pred),
        'token_size': config['bind'],
        'test_size': [len(y_pred)]*len(y_pred),
        'person': y_names,
        'true' : y_test,
        'pred' : y_pred,
    }


    save_time = datetime.now().__format__("%m%d_%H%M%S%Z")
    pd.DataFrame(val_df).to_csv(f'_{model}_.csv',index=False)  
    pd.DataFrame(pred_dict).to_csv(f'_{model}_pred.csv',index=False)



# confusion matrix graph
def plot_confusion_matrix__(conf_matrix, y_pred, y_test, model_name):

    conf_matrix_percentage = conf_matrix / np.sum(conf_matrix) * 100
    plt.rcParams['font.family'] = 'Comic Sans MS'


    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.1) 


    annot_array = np.array([
        ["{0:.2f}%\n({1})".format(pct, value) for pct, value in zip(row_percentage, row_values)] 
        for row_percentage, row_values in zip(conf_matrix_percentage, conf_matrix)
    ])

    sns.heatmap(conf_matrix, annot=annot_array, fmt='', cmap='Reds', annot_kws={"size": 12}, cbar=False, 
                xticklabels=True, yticklabels=True, square=True, linewidths=1, linecolor='grey')
    plt.xlabel('Predicted labels', fontsize=14)
    plt.ylabel('True labels', fontsize=14)
    plt.title(f'Confusion Matrix for {model_name}', fontsize=16)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()  # Adjusts the plot to ensure everything fits without overlap
    plt.show()

