#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 16:58:52 2020

@author: em812
"""
from pathlib import Path
import pandas as pd
from tierpsytools.preprocessing.scaling_class import scalingClass
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from tierpsytools.analysis.cv_splitters import StratifiedGroupKFold
from tierpsytools.analysis.scorers import ClassifScorer
from feat_selection import main_feature_selection
from optimize_hyperparameters import main_optimize_hyperparameters
from predict_test_set import main_predict_test_set
import pdb
from moaclassification import INPUT_DIR

#%%
# Input parameters
align_blue = True
balance_classes = True
n_average = 20
scale = 'rescale1'
scaler = scalingClass(scaling=scale)

# Estimator parameters
estimator = LogisticRegression(
    penalty='elasticnet', l1_ratio=0.5, C=1, solver='saga',
    multi_class='multinomial', max_iter=500)

pipe = Pipeline([
    ('scaler', scaler), ('estimator', estimator)
    ])

# CV parameters
n_folds = 4
cv = StratifiedGroupKFold(n_splits=n_folds, random_seed=724)
vote_type = 'counts'
scorer = ['accuracy', ClassifScorer(scorer='f1',average='macro'),
          ClassifScorer(scorer='roc_auc', multi_class='ovo', average='macro'),
          'mcc']
scorenames = ['accuracy', 'f1', 'roc_auc', 'mcc']

# Define input directories
root_in = Path(INPUT_DIR) / 'get_smoothed_balanced_data' / 'smoothed_balanced_data'

# Input directory
data_file = root_in / 'features_cv.csv'
meta_file = root_in / 'metadata_cv.csv'

# Save results to
saveto = Path().cwd() / 'results'
saveto.mkdir(parents=True, exist_ok=True)


#%% Read and preprocess data
feat = pd.read_csv(data_file, index_col=None)
meta = pd.read_csv(meta_file, index_col=None)

# Remove DMSO points
meta = meta[meta['MOA_group']!=0.0]
feat = feat.loc[meta.index]

#%% Feature selection
best_feat_set = main_feature_selection(
    feat, meta['MOA_group'].values, meta['drug_type'].values, meta['drug_dose'].values,
    scaler, estimator, cv, vote_type, scorer, scorenames, saveto)

#%% Optimize hyperparameters
best_estimator = main_optimize_hyperparameters(
    feat[best_feat_set], meta['MOA_group'].values, meta['drug_type'].values,
    meta['drug_dose'].values, pipe, cv, vote_type,
    scorer, scorenames, saveto)

#%% Read test set
feat_test = pd.read_csv(str(data_file).replace('_cv','_test'), index_col=None)
meta_test = pd.read_csv(str(meta_file).replace('_cv','_test'), index_col=None)

# remove DMSO
meta_test = meta_test[meta_test['MOA_group']!=0.0]
feat_test = feat_test.loc[meta_test.index]

#%% Classification of test set
main_predict_test_set(
    feat_test[best_feat_set], meta_test['MOA_group'].values,
    meta_test['drug_type'].values, meta_test['drug_dose'].values,
    best_estimator, cv, vote_type, scorer, scorenames, saveto)


