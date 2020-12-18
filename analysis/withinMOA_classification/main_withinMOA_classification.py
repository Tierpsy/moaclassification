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
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from tierpsytools.analysis.cv_splitters import StratifiedGroupKFold
from tierpsytools.analysis.scorers import ClassifScorer
from tierpsytools.analysis.classification_tools import plot_confusion_matrix
from tierpsytools.analysis.classification_tools import cv_predict
from tierpsytools.analysis.helper import _get_multi_sclassifscorers
import pickle
from moaclassification import INPUT_DIR, ANALYSIS_DIR

N_GROUPS_PER_MOA = \
    {2: 3, 3: 2, 6: 3, 7: 3, 9: 2, 10: 2, 11: 2, 13: 4, 15: 3, 17: 3}

def _compile_results(drug_type, drug_dose, pred, probas, labels, saveto):
    """
    Local function to compile and save results for a signle moa
    """
    res = pd.DataFrame({
        'drug_type': drug_type,
        'drug_dose': drug_dose,
        'prediction': pred
        })
    res = pd.concat([
        res, pd.DataFrame(probas, columns=labels)], axis=1)

    pickle.dump(res, open(saveto, 'wb'))

    return res

#%% Input

# Data
data_file = Path(INPUT_DIR) / 'features.csv'
meta_file = Path(INPUT_DIR) / 'metadata.csv'

# test-train split files
train_file = Path(INPUT_DIR) / 'split' / 'train_compounds.csv'
test_file = Path(INPUT_DIR) / 'split' / 'test_compounds.csv'

# moa info file
moa_file = Path(INPUT_DIR) / 'AllCompoundsMoA.csv'

# CV
n_folds = 4
splitter_grouped = StratifiedGroupKFold(n_splits=n_folds, random_seed=724)
splitter = StratifiedKFold(n_splits=n_folds)

# Scores
scorer = ['accuracy', ClassifScorer(scorer='f1',average='macro')]
scorenames = ['accuracy', 'f1']

# Feature set
root_classif = Path(ANALYSIS_DIR) / 'classification' / 'results'
path_to_feat_set = root_classif / 'feature_selection_results' / 'best_feature_set.csv'
feat_set = pd.read_csv(path_to_feat_set, header=None)[0].to_numpy()

# Estimator parameters
path_to_params = root_classif / 'optimize_hyperparameters_results' / 'best_params.p'
params = pickle.load(open(path_to_params, 'rb'))

# Estimator
estimator = LogisticRegression(**params)

scale = 'rescale1'

# Save path
saveroot = Path().cwd() / 'results'
saveroot.mkdir(parents=True, exist_ok=True)

#%% Read and preprocess data
X = pd.read_csv(data_file, index_col=None)[feat_set]
meta = pd.read_csv(meta_file, index_col=None)
moa_info = pd.read_csv(moa_file, index_col=None)

# Get names of compounds
meta['drug_name'] = meta['drug_type'].map(dict(moa_info[['CSN', 'drug_name']].values))

# Keep only cv and test set
all_compounds = \
    pd.read_csv(train_file, usecols=['drug_type'])['drug_type'].to_list() + \
    pd.read_csv(test_file, usecols=['drug_type'])['drug_type'].to_list()

X = X[meta['drug_type'].isin(all_compounds)]
meta = meta[meta['drug_type'].isin(all_compounds)]

# Impute nans
X = X.fillna(X.mean())

# Get class labels
y = meta[['drug_name', 'MOA_specific']].apply(
    lambda x: '_'.join(x.astype(str).values), axis=1).values
# Crop class labels to 35 characters
y = np.array([x[:max(len(x), 35)] for x in y])

# Get scorers in a dictionary
scorers = _get_multi_sclassifscorers(scorer)

#%% Classify in compounds
#----------------------------
# Get mapper from moa id to moa names
moa2name = dict(meta[['MOA_group','MOA_general']].values)

# Loop over modes of action
for moa, n_groups in N_GROUPS_PER_MOA.items():

    print('Predicting for MOA {}'.format(moa))

    saveto = saveroot / 'MOA='.format(moa)
    saveto.mkdir(exist_ok=True)

    mask = meta['MOA_group']==moa

    # Cross-validation predictions (to make confusion matrix)
    _pred, _probas, _labels, _test_folds, _ = cv_predict(
        X.loc[mask], y[mask], splitter, estimator, scale_function=scale,
        n_jobs=1)

    # Compute scores
    _scores = {}
    for scorename in scorers.keys():
        _scores[scorename] = []
        for test_index in _test_folds:
            _scores[scorename].append(
                scorers[scorename].score(
                    y[test_index], pred=_pred[test_index],
                    probas=_probas[test_index], labels=_labels
                    ))

    res = _compile_results(y[mask], meta.loc[mask, 'drug_dose'], _pred, _probas, _labels, saveto/'res.p')
    pickle.dump(_scores, open(saveto/'scores.p', 'wb'))

    # plot the confusion matrix for the given mode of action
    plot_confusion_matrix(
        y, _pred, classes=None, normalize=False,
        title=moa2name[moa]+'\ncross validation accuracy = {:.1f}%'.format(
            np.mean(_scores['accuracy'])*100),
        figsize=(8,8), cmap=None, cluster=True, n_clusters=n_groups,
        saveto=saveto/'confusion_matrix_MOA={}.svg'.format(moa))

