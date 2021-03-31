#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification with randomly selected feature sets to estimate the performance
of stacking compared to individual strains and pooling strains together

Created on Thu Jul 30 11:12:48 2020

@author: em812
"""

from pathlib import Path
import pandas as pd
from moaclassification.helper import get_drug2moa_mapper
from moaclassification import INPUT_DIR
from sklearn.model_selection import StratifiedKFold, train_test_split

#%% Input
random_state = 6767
align_blue = True
feat_set = 'all'

n_average_doses = 5
balance = True

# root directory
root_in = Path(INPUT_DIR)

# features and metadata files
feat_file = root_in/ 'features.csv'
meta_file = root_in/ 'metadata.csv'

saveto = Path().cwd() / \
    'filtered_align_blue={}'.format(
        align_blue, balance, n_average_doses)
saveto.mkdir(parents=True, exist_ok=True)

#%% Read and preprocess data
feat = pd.read_csv(feat_file)
meta = pd.read_csv(meta_file)

mask = meta['drug_type'] == 'DMSO'
feat = feat[mask]
meta = meta[mask]

meta = meta.reset_index(drop=True)
feat = feat.reset_index(drop=True)

#%% Make fake moa labels
moa_maker = StratifiedKFold(n_splits=10)

for i,(_, idx) in enumerate(moa_maker.split(feat, meta['date_yyyymmdd'])):
    meta.loc[idx, 'MOA_group'] = i
    meta.loc[idx, 'MOA_general'] = i
    meta.loc[idx, 'MOA_specific'] = i

meta['drug_type'] = list(range(meta.shape[0]))

#%% Preprocess
# CV-test split
feat_cv, feat_test, meta_cv, meta_test = \
    train_test_split(feat, meta, test_size=0.2, stratify=meta['MOA_group'],
                     random_state=random_state)

# Impute nans
feat_cv = feat_cv.fillna(feat_cv.mean())
feat_test = feat_test.fillna(feat_cv.mean())


#%% Store
feat_cv.to_csv(saveto/'features_cv.csv', index=None)
feat_test.to_csv(saveto/'features_test.csv', index=None)
meta_cv.to_csv(saveto/'metadata_cv.csv', index=None)
meta_test.to_csv(saveto/'metadata_test.csv', index=None)
