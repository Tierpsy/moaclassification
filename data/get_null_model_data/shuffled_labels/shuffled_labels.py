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

# test_train split directory
root_split = root_in / 'split'

# features and metadata files
feat_file = root_in/ 'features.csv'
meta_file = root_in/ 'metadata.csv'

# test-train split files
train_file = root_split / 'train_compounds.csv'
test_file = root_split / 'test_compounds.csv'

saveto = Path().cwd() / \
    'filtered_align_blue={}'.format(
        align_blue, balance, n_average_doses)
saveto.mkdir(parents=True, exist_ok=True)

#%% Read and preprocess data
feat = pd.read_csv(feat_file)
meta = pd.read_csv(meta_file)

# Remove DMSO
mask = meta['drug_type']!='DMSO'
feat = feat[mask]
meta = meta[mask]

# Keep only the compounds with siginificant effect in the train/test dataset
train_comp = pd.read_csv(train_file)
test_comp = pd.read_csv(test_file)

mask = meta['drug_type'].isin(
    train_comp['drug_type'].to_list() + test_comp['drug_type'].to_list())
feat = feat[mask]
meta = meta[mask]

#%% Shuffle drug moa labels
drug_moas = meta[['MOA_group', 'MOA_general', 'MOA_specific', 'drug_type']].drop_duplicates(subset=['drug_type'])
drug_moas['drug_type'] = drug_moas['drug_type'].sample(frac=1, replace=False).values
drug_moas = drug_moas.set_index('drug_type')
for drug in meta['drug_type'].unique():
    mask = meta['drug_type']==drug
    meta.loc[mask, ['MOA_group', 'MOA_general', 'MOA_specific']] = drug_moas.loc[[drug]*mask.sum()].values

#%%  CV-test split
drug_moas = drug_moas.reset_index(drop=False)
drugs_cv, drugs_test = \
    train_test_split(drug_moas, test_size=0.2, stratify=drug_moas['MOA_group'],
                     random_state=random_state)
cv_ids = meta['drug_type'].isin(drugs_cv['drug_type'])
test_ids = meta['drug_type'].isin(drugs_test['drug_type'])

feat_cv = feat[cv_ids]
meta_cv = meta[cv_ids]
feat_test = feat[test_ids]
meta_test = meta[test_ids]

#%% Impute nans
feat_cv = feat_cv.fillna(feat_cv.mean())
feat_test = feat_test.fillna(feat_cv.mean())


#%% Store
feat_cv.to_csv(saveto/'features_cv.csv', index=None)
feat_test.to_csv(saveto/'features_test.csv', index=None)
meta_cv.to_csv(saveto/'metadata_cv.csv', index=None)
meta_test.to_csv(saveto/'metadata_test.csv', index=None)
