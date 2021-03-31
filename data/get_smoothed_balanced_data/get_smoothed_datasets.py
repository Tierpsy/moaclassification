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
from tierpsytools.analysis.drug_screenings.bagging_drug_data import \
    DrugDataBagging,DrugDataBaggingByMOA
from moaclassification.helper import get_drug2moa_mapper
from moaclassification.preprocessing import cv_test_split, impute_nans
from tierpsytools.feature_processing.preprocess_features import encode_categorical_variable
from moaclassification import INPUT_DIR
import pdb

#%% Input
random_state = 6767
align_blue = True
average_dose = False
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
novelty_file = root_split / 'novelty_compounds.csv'

saveto = Path().cwd() / \
    'filtered_align_blue={}_balanced={}_n_average={}'.format(
        align_blue, balance, n_average_doses)
saveto.mkdir(parents=True, exist_ok=True)

#%% Read and preprocess data
feat = pd.read_csv(feat_file)
meta = pd.read_csv(meta_file)

#%% MOA mapper
mapper_moa = get_drug2moa_mapper(meta['drug_type'].values, meta['MOA_group'].values)
mapper_moa_gen = get_drug2moa_mapper(meta['drug_type'].values, meta['MOA_general'].values)
mapper_moa_spec = get_drug2moa_mapper(meta['drug_type'].values, meta['MOA_specific'].values)

#%% Preprocess
feat_cv, meta_cv, feat_test, meta_test = cv_test_split(feat, meta, train_file, test_file)

novelty_compounds = pd.read_csv(novelty_file)
feat_nov = feat[meta['drug_type'].isin(novelty_compounds['drug_type'])]
meta_nov = meta[meta['drug_type'].isin(novelty_compounds['drug_type'])]

if align_blue:
    feat_cv = feat_cv.fillna(feat_cv.mean())
    feat_test = feat_test.fillna(feat_cv.mean())
    feat_nov = feat_nov.fillna(feat_cv.mean())
else:
    means_cv = {blue: x.mean() for blue,x in feat_cv.groupby(by=meta_cv['bluelight'])}
    feat_cv = [x.fillna(means_cv[blue])
            for blue,x in feat_cv.groupby(by=meta_cv['bluelight'])]
    feat_cv = pd.concat(feat_cv).sort_index()
    feat_test = [x.fillna(means_cv[blue])
            for blue,x in feat_test.groupby(by=meta_test['bluelight'])]
    feat_test = pd.concat(feat_test).sort_index()
    feat_nov = [x.fillna(means_cv[blue])
            for blue,x in feat_nov.groupby(by=meta_nov['bluelight'])]
    feat_nov = pd.concat(feat_nov).sort_index()

#%% Get augmented average doses
print('Getting augmented data...')
for lab, feat, meta, balance_classes in [
        ('cv', feat_cv, meta_cv, balance),
        ('test', feat_test, meta_test, False),
        ('novel', feat_nov, meta_nov, False)
        ]:
    if balance_classes:
        ndrugs_per_moa = meta[['MOA_group', 'drug_type']].groupby(by=['MOA_group'])['drug_type'].nunique()
        frac_per_moa = ndrugs_per_moa.max() / ndrugs_per_moa

        if align_blue:
            bagger = DrugDataBaggingByMOA(
                multiplier=n_average_doses, replace=True, frac_per_dose=0.6,
                random_state=random_state, bluelight_conditions=False)
            X, meta1 = \
                bagger.fit_transform(feat, meta['MOA_group'], meta['drug_type'].values,
                                     meta['drug_dose'], shuffle=True)
        else:
            bagger = DrugDataBaggingByMOA(
                multiplier=n_average_doses, replace=True, frac_per_dose=0.6,
                random_state=random_state, bluelight_conditions=True)
            X, meta1 = \
                bagger.fit_transform(feat, meta['MOA_group'], meta['drug_type'].values, meta['drug_dose'],
                                     bluelight=meta['bluelight'], shuffle=True)
    else:
        if align_blue:
            bagger = DrugDataBagging(
                n_bags=n_average_doses, replace=True, frac_per_dose=0.6,
                random_state=random_state, average_dose=True, bluelight_conditions=False)
            X, meta1 = \
                bagger.fit_transform(feat, meta['drug_type'].values, meta['drug_dose'], shuffle=True)
        else:
            bagger = DrugDataBagging(
                n_bags=n_average_doses, replace=True, frac_per_dose=0.6,
                random_state=random_state, average_dose=True, bluelight_conditions=True)
            X, meta1 = \
                bagger.fit_transform(feat, meta['drug_type'].values, meta['drug_dose'],
                                     bluelight=meta['bluelight'], shuffle=True)
        X = pd.concat(X, axis=0)
        meta1 = pd.concat(meta1, axis=0)

    meta1['MOA_group'] = meta1['drug_type'].map(mapper_moa).values
    meta1['MOA_general'] = meta1['drug_type'].map(mapper_moa_gen).values
    meta1['MOA_specific'] = meta1['drug_type'].map(mapper_moa_spec).values

    if not align_blue:
        X = encode_categorical_variable(X, meta1['bluelight'])

    # Store
    X.to_csv(saveto/'augmented_average_dose_dataset_{}.csv'.format(lab), index=None)
    meta1.to_csv(saveto/'augmented_average_dose_metadata_{}.csv'.format(lab), index=None)
