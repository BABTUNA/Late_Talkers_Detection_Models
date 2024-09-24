import pandas as pd
import numpy as np
import torch
import os 

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

from utilis import plot_confusion_matrix__
from utilis import save_results
from utilis import save_results_

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


import spacy
import string



# run random forest, decision tree, logistic regression, and ada boost models
# run on k splits on selected dataset with a single modality feature
# take the average of each k split for model performance

def run_tests(df_data, feature, splits, random_state):

    # ml_results will contain all results from all models (takes average of each split)
    ml_results = pd.DataFrame()

    # create groups and stratify data in splits based off name
    groups = df_data['name'].tolist()
    group_kfold = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=random_state)


    # run decision tree model with k splits
    total  = 0 
    dt_results = []
    dt_model = DecisionTreeClassifier(random_state=random_state)
    for train_idxs, test_idxs in group_kfold.split(df_data[feature], df_data['label'], groups):#for i, (train_idxs, test_idxs) in enumerate(kf.split(df_data, df_data['name'])):
        df_data['split'] = ['train' if idx in train_idxs else 'test' for idx in range(len(df_data))]
        X_train = df_data[df_data['split'] == 'train'][feature].tolist()  # Ensure 'feature_scaled' is your feature column
        y_train = df_data[df_data['split'] == 'train']['label'].tolist()
        X_test = df_data[df_data['split'] == 'test'][feature].tolist()
        y_test = df_data[df_data['split'] == 'test']['label'].tolist()
        train_names = df_data[df_data['split'] == 'train']['name'].tolist()
        test_names = df_data[df_data['split'] == 'test']['name'].tolist()
        
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)
        # print(len(y_test), len(y_pred))
        # print(y_test)
        # print("split")
        # print(y_pred)

        report = classification_report(y_test, y_pred, output_dict=True)
        unique_classes = sorted(set(y_test))
        # print([str(c) for c in unique_classes])
        metrics_dict = classification_report(
            y_test, 
            y_pred, 
            zero_division=1, 
            labels=[0, 1], 
            target_names=['0', '1'], 
            output_dict=True
        )

        df_result = pd.DataFrame(metrics_dict).transpose()
        df_result = df_result.map(lambda x: f" {x} " if isinstance(x, (int, float)) else x)
        df_result = df_result.reset_index()
        df_result.columns = ['data','precision', 'recall', 'f1-score', 'support']
        dt_results.append(df_result)

        cm = confusion_matrix(y_test, y_pred)

        if len(cm.ravel()) == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  


        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        total += specificity

    total_spec = total/splits
    combined_df = pd.concat(dt_results, ignore_index=True)
    numeric_columns = ['precision', 'recall', 'f1-score', 'support']
    for column in numeric_columns:
        combined_df[column] = pd.to_numeric(combined_df[column], errors='coerce')

    average_df = combined_df.groupby('data', as_index=False)[numeric_columns].mean()
    average_df['specificity'] = total_spec

    blank_row = pd.DataFrame({col: ['-'] for col in average_df.columns})

    ml_results= pd.concat([ml_results, blank_row , average_df], axis=0, ignore_index=True)
    

      # run random forest model with k splits
    total  = 0 
    dt_results = []
    dt_model = RandomForestClassifier(random_state=random_state)
    for train_idxs, test_idxs in group_kfold.split(df_data[feature], df_data['label'], groups):#for i, (train_idxs, test_idxs) in enumerate(kf.split(df_data, df_data['name'])):
        df_data['split'] = ['train' if idx in train_idxs else 'test' for idx in range(len(df_data))]
        X_train = df_data[df_data['split'] == 'train'][feature].tolist()  # Ensure 'feature_scaled' is your feature column
        y_train = df_data[df_data['split'] == 'train']['label'].tolist()
        X_test = df_data[df_data['split'] == 'test'][feature].tolist()
        y_test = df_data[df_data['split'] == 'test']['label'].tolist()
        train_names = df_data[df_data['split'] == 'train']['name'].tolist()
        test_names = df_data[df_data['split'] == 'test']['name'].tolist()
        
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)


        report = classification_report(y_test, y_pred, output_dict=True)
        unique_classes = sorted(set(y_test))
        metrics_dict = classification_report(
            y_test, 
            y_pred, 
            zero_division=1, 
            labels=[0, 1],  # Specify both classes, even if one is missing in y_test
            target_names=['0', '1'], 
            output_dict=True
        )

        df_result = pd.DataFrame(metrics_dict).transpose()
        df_result = df_result.map(lambda x: f" {x} " if isinstance(x, (int, float)) else x)
        df_result = df_result.reset_index()
        df_result.columns = ['data','precision', 'recall', 'f1-score', 'support']
        dt_results.append(df_result)

        cm = confusion_matrix(y_test, y_pred)

        if len(cm.ravel()) == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  # Handle the case with only one class

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        total += specificity

    total_spec = total/splits
    combined_df = pd.concat(dt_results, ignore_index=True)
    numeric_columns = ['precision', 'recall', 'f1-score', 'support']
    for column in numeric_columns:
        combined_df[column] = pd.to_numeric(combined_df[column], errors='coerce')

    average_df = combined_df.groupby('data', as_index=False)[numeric_columns].mean()
    average_df['specificity'] = total_spec

    blank_row = pd.DataFrame({col: ['-'] for col in average_df.columns})
    ml_results = pd.concat([ml_results,blank_row , average_df], axis=0, ignore_index=True)




    # run logistic regression model with k splits
    total  = 0 
    dt_results = []
    dt_model = LogisticRegression(random_state=random_state)
    for train_idxs, test_idxs in group_kfold.split(df_data[feature], df_data['label'], groups):#for i, (train_idxs, test_idxs) in enumerate(kf.split(df_data, df_data['name'])):
        df_data['split'] = ['train' if idx in train_idxs else 'test' for idx in range(len(df_data))]
        X_train = df_data[df_data['split'] == 'train'][feature].tolist()  # Ensure 'feature_scaled' is your feature column
        y_train = df_data[df_data['split'] == 'train']['label'].tolist()
        X_test = df_data[df_data['split'] == 'test'][feature].tolist()
        y_test = df_data[df_data['split'] == 'test']['label'].tolist()
        train_names = df_data[df_data['split'] == 'train']['name'].tolist()
        test_names = df_data[df_data['split'] == 'test']['name'].tolist()
        
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)

        # print(len(y_test), len(y_pred))
        # print(y_test)
        # print("split")
        # print(y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        unique_classes = sorted(set(y_test))
        metrics_dict = classification_report(
            y_test, 
            y_pred, 
            zero_division=1, 
            labels=[0, 1],  # Specify both classes, even if one is missing in y_test
            target_names=['0', '1'], 
            output_dict=True
        )

        df_result = pd.DataFrame(metrics_dict).transpose()
        df_result = df_result.map(lambda x: f" {x} " if isinstance(x, (int, float)) else x)
        df_result = df_result.reset_index()
        df_result.columns = ['data','precision', 'recall', 'f1-score', 'support']
        dt_results.append(df_result)

        cm = confusion_matrix(y_test, y_pred)

        if len(cm.ravel()) == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  # Handle the case with only one class

        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        total += specificity

    total_spec = total/splits
    combined_df = pd.concat(dt_results, ignore_index=True)
    numeric_columns = ['precision', 'recall', 'f1-score', 'support']
    for column in numeric_columns:
        combined_df[column] = pd.to_numeric(combined_df[column], errors='coerce')

    average_df = combined_df.groupby('data', as_index=False)[numeric_columns].mean()
    average_df['specificity'] = total_spec

    blank_row = pd.DataFrame({col: ['-'] for col in average_df.columns})
    ml_results = pd.concat([ml_results,blank_row , average_df], axis=0, ignore_index=True)


    # run ada boost model with k splits

    total  = 0 
    dt_results = []
    dt_model = AdaBoostClassifier(random_state=random_state)
    for train_idxs, test_idxs in group_kfold.split(df_data[feature], df_data['label'], groups):#for i, (train_idxs, test_idxs) in enumerate(kf.split(df_data, df_data['name'])):
        df_data['split'] = ['train' if idx in train_idxs else 'test' for idx in range(len(df_data))]
        X_train = df_data[df_data['split'] == 'train'][feature].tolist()  # Ensure 'feature_scaled' is your feature column
        y_train = df_data[df_data['split'] == 'train']['label'].tolist()
        X_test = df_data[df_data['split'] == 'test'][feature].tolist()
        y_test = df_data[df_data['split'] == 'test']['label'].tolist()
        train_names = df_data[df_data['split'] == 'train']['name'].tolist()
        test_names = df_data[df_data['split'] == 'test']['name'].tolist()
        
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)


        report = classification_report(y_test, y_pred, output_dict=True)
        unique_classes = sorted(set(y_test))
        metrics_dict = classification_report(
            y_test, 
            y_pred, 
            zero_division=1, 
            labels=[0, 1],  # Specify both classes, even if one is missing in y_test
            target_names=['0', '1'], 
            output_dict=True
        )

        df_result = pd.DataFrame(metrics_dict).transpose()
        df_result = df_result.map(lambda x: f" {x} " if isinstance(x, (int, float)) else x)
        df_result = df_result.reset_index()
        df_result.columns = ['data','precision', 'recall', 'f1-score', 'support']
        dt_results.append(df_result)

        cm = confusion_matrix(y_test, y_pred)

        if len(cm.ravel()) == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]  # Handle the case with only one class
            
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        total += specificity

    total_spec = total/splits
    combined_df = pd.concat(dt_results, ignore_index=True)
    numeric_columns = ['precision', 'recall', 'f1-score', 'support']
    for column in numeric_columns:
        combined_df[column] = pd.to_numeric(combined_df[column], errors='coerce')

    average_df = combined_df.groupby('data', as_index=False)[numeric_columns].mean()
    average_df['specificity'] = total_spec

    blank_row = pd.DataFrame({col: ['-'] for col in average_df.columns})
    ml_results = pd.concat([ml_results,blank_row , average_df], axis=0, ignore_index=True)


    # return ml result

    return ml_results