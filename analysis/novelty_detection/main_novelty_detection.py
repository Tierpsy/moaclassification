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
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from tierpsytools.analysis.cv_splitters import StratifiedGroupKFold
from tierpsytools.analysis.scorers import ClassifScorer
from tierpsytools.analysis.classification_tools import cv_predict
from tierpsytools.preprocessing.scaling_class import scalingClass
from moaclassification.helper import apply_mask
from moaclassification.novelty_detection_fun import \
    get_theta_scores_in, get_theta_scores_out, get_novelty_score
from moaclassification import INPUT_DIR, ANALYSIS_DIR
import pdb

#%% Input
scale = 'rescale1'
novelty_from = 'probas' # 'votes'

# Input directory
root_in = Path(INPUT_DIR) / 'get_smoothed_balanced_data' / 'smoothed_balanced_data'

data_file = root_in / 'features_cv.csv'
meta_file = root_in/ 'metadata_cv.csv'

moa_file = Path(INPUT_DIR) / 'AllCompoundsMoA.csv'

# CV
n_folds = 4
splitter = StratifiedGroupKFold(n_splits=n_folds, random_seed=724)

# Scores
scorer = ['accuracy', ClassifScorer(scorer='f1',average='macro')]
scorenames = ['accuracy', 'f1']

# Feature set
root_classif = Path(ANALYSIS_DIR) / 'classification' / 'results'
path_to_set = root_classif / 'feature_selection_results' / 'best_feature_set.csv'
feat_set = pd.read_csv(path_to_set, header=None)[0].to_numpy()

# Base estimator pipeline
# (estimator used to classify tierpsy feature vectors to modes of action)
path_to_params = root_classif / 'optimize_hyperparameters_results' / 'best_params.p'
params = pickle.load(open(path_to_params, 'rb'))

estimator = LogisticRegression(**params)

pipe = Pipeline([('scaler', scalingClass(scaling=scale)),
                    ('estimator', estimator)])

# Ensemble of binary estimators pipeline
# (estimators used to classify compounds in known-novel based on theta scores)
svc = SVC(C=1, kernel='linear', class_weight='balanced', gamma='auto')
binary_pipe = Pipeline([('scaler', StandardScaler()), ('svc', svc)])

saveto = Path().cwd() / 'results'
saveto.mkdir(parents=True, exist_ok=True)

#%% Read data
feat = pd.read_csv(data_file, usecols=feat_set)
y = pd.read_csv(meta_file, usecols=['MOA_group']).values.flatten()
group = pd.read_csv(meta_file, usecols=['drug_type']).values.flatten()

feat, y, group = apply_mask(group!='DMSO', [feat, y, group])

# Novel compounds
feat_nov = pd.read_csv(str(data_file).replace('cv', 'novel'), usecols=feat_set)
y_nov = pd.read_csv(str(meta_file).replace('cv', 'novel'),
                    usecols=['MOA_group']).values.flatten()
group_nov = pd.read_csv(str(meta_file).replace('cv', 'novel'),
                        usecols=['drug_type']).values.flatten()

# Test compounds
feat_test = pd.read_csv(str(data_file).replace('cv', 'test'), usecols=feat_set)
y_test = pd.read_csv(str(meta_file).replace('cv', 'test'),
                     usecols=['MOA_group']).values.flatten()
group_test = pd.read_csv(str(meta_file).replace('cv', 'test'),
                         usecols=['drug_type']).values.flatten()

# Make mapper
moa_info = pd.read_csv(moa_file)
mapper = dict(moa_info[['CSN', 'MOA_group']].drop_duplicates(subset='drug_type').values)


#%%
##===========================
##%% PART 1:
##===========================
## Using the cv dataset, we partition the data in known classes and
## presumed-novel classes.
## We get the theta scores from each partition and train binary svms
## that flag compounds as known or novel, based on their theta scores.

#%% Partition cv data labeling one class at a time as presumed-novel:
### Run CV for each partition and get theta scores for compounds of known
### classes and theta scores of compounds of the presumed-novel class

# Initialize dictionary for thetas of known classes
thetas_in = {}
# Initialize dictionary for thetas of presumed-novel class
thetas_out = {}
# Initialize dictionary for average thetas of known classes
thetas_i = {}
for moa in np.unique(y):
    mask = (y != moa).values

    # CV
    pred, probas, labels, test_folds, trained_estimators = cv_predict(
        feat.loc[mask, :], y[mask], splitter, pipe, group=group[mask],
        n_jobs=-1, sample_weight=None)

    probas_in = pd.DataFrame(probas, columns=labels, index=group[mask])

    # Get theta scores for in drugs (compounds of known classes)
    thetas_in[moa], thetas_i[moa] = get_theta_scores_in(
        probas_in, probas_in.index, probas_in.columns, group2moa_mapper=mapper)

    # Train estimator with all the known classes to predict presumed-novel
    pipe.fit(feat.loc[mask, :], y[mask])
    probas_out = pd.DataFrame(
        pipe.predict_proba(feat.loc[~mask, :]), columns=labels, index=group[~mask])

    # Get theta scores for out drugs (compounds of presumed-novel class)
    thetas_out[moa] = get_theta_scores_out(
        probas_out, probas_out.index,
        probas_out.columns, thetas_i[moa])

#%% Train SVMs with thetas: one SVM per partition
svm_feats = ['theta_score', 'ratio'] #'moa_theta_i',

# build a dictionary with the fitted binary classifiers and their respective
# cv accuracy
svms = {}
for moa in np.unique(y):

    X = pd.concat([
        thetas_in[moa].loc[thetas_in[moa]['is_correct_class'].values, svm_feats], #
        thetas_out[moa][svm_feats]
        ])
    label = np.concatenate([
        np.zeros(thetas_in[moa]['is_correct_class'].sum()), #
        np.ones(thetas_out[moa].shape[0])
        ])

    res = cross_validate(binary_pipe, X, label, cv=4, n_jobs=-1)
    #print('CV accuracy = {}'.format(res['test_score']))

    svms[moa] = {'estimator': binary_pipe.fit(X, label),
                 'weight': np.mean(res['test_score'])}


#%%
##===========================
##%% PART 2:
##===========================
## Using the trained binary classifiers in 'svms', we get the novelty score
## for the compounds of the test set and the novel compounds

#%% Train a multiclass model with the entire training set
### and predict test and novel compounds
pipe.fit(feat, y)

probas_nov_all = pd.DataFrame(
    pipe.predict_proba(feat_nov),
    columns=pipe['estimator'].classes_, index=group_nov)

probas_test_all = pd.DataFrame(
    pipe.predict_proba(feat_test),
    columns=pipe['estimator'].classes_, index=group_test)

pickle.dump(pipe, open(saveto/'fitted_pipe_model-all_moas.p', 'wb'))

#%% Get the theta values for the novel compounds
thetas_nov = {}
for moa in np.unique(y):
    thetas_nov[moa] = get_theta_scores_out(
        probas_nov_all, probas_nov_all.index,
        probas_nov_all.columns, thetas_i[moa])

#%% Get the theta values for the test compounds
thetas_test = {}
for moa in np.unique(y):
    thetas_test[moa] = get_theta_scores_out(
        probas_test_all, probas_test_all.index,
        probas_test_all.columns, thetas_i[moa])

#%% Get novelty score of novel compounds
novelty_score = get_novelty_score(svms, thetas_nov, svm_feats)

# Save
pd.DataFrame(novelty_score, columns=['novelty_score']).to_csv(
    saveto/'novelty_scores_novel.csv')

#%% Get novelty score of test compounds
novelty_score_test = get_novelty_score(svms, thetas_test, svm_feats)

# Get the test predictions and real class labels to add to the results
y_test = pd.Series(y_test).groupby(by=group_test).mean()
y_test = y_test.loc[thetas_test[moa].index]

novelty_score_test = pd.DataFrame(novelty_score_test, columns=['novelty_score'])

novelty_score_test = novelty_score_test.assign(
    y_true=y_test.loc[novelty_score_test.index])
novelty_score_test = novelty_score_test.assign(
    y_pred=thetas_test[moa].loc[novelty_score_test.index, 'most_likely_class'])

# Store results
novelty_score_test.to_csv(saveto/'novelty_scores_test.csv')

