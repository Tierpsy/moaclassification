#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification with randomly selected feature sets to estimate the performance
of stacking compared to individual strains and pooling strains together

Created on Thu Jul 30 11:12:48 2020

@author: em812
"""

import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from tierpsytools.analysis.drug_screenings.MIL.majority_vote import \
    majority_vote_CV
from moaclassification.classification_helper import \
    make_score_df, make_pred_df, _plot_confusion_matrix
import pdb

PARAM_GRID = {
    'estimator__penalty': ['l1', 'l2'],
    'estimator__C': [10**i for i in range(-2,4)],
    'estimator__multi_class': ['multinomial', 'ovr'],
    'estimator__solver': ['saga'],
    'estimator__max_iter': [500]
    }
N_ITER = 20

#%% Main
def main_optimize_hyperparameters(
        feat, y, group, dose,
        pipeline, cv,
        vote_type, scorer, scorenames,
        saveroot):

    saveto = saveroot / 'optimize_hyperparameters_results'
    saveto.mkdir(parents=True, exist_ok=True)

    ## Use the random grid to search for best hyperparameters
    # Random search of parameters, using 4 fold cross validation,
    # search across N_ITER different combinations.
    # The hyperparameter selection is done based on classification accuracy
    # of individual replicates (not compound-level predictions).
    # Ideally, we would want to select the parameters based the compound-level
    # classification accuracy. However, selecting based on replicate-level
    # accuracy is accepted as a viable alternative, since the two scores are
    # highly correlated and in most cases the higest compound-level accuracy
    # is achieved when we have the highest replicate-level accurecy.
    #------------------------------
    rf_random = RandomizedSearchCV(
        estimator = clone(pipeline), param_distributions = PARAM_GRID,
        n_iter=N_ITER, cv=4, random_state=42, refit=True, n_jobs=-1)

    # Fit the random search model
    rf_random.fit(feat, y)

    pickle.dump(rf_random, open(saveto/'fitted_RandomizedSearchCV.p', 'wb'))
    pickle.dump(rf_random.best_estimator_['estimator'].get_params(),
                open(saveto/'best_params.p', 'wb'))

    # Get the majority vote for the best hyperparameters
    _score, _score_maj, _pred, _probas, labels =  majority_vote_CV(
        feat, y, group, clone(rf_random.best_estimator_), cv,
        vote_type=vote_type, scale_function=None, n_jobs=-1,
        scorer=scorer, return_predictions=True)

    ## Store results
    #------------------------------
    # All CV scores
    score_df = make_score_df(_score, _score_maj)
    score_df.to_csv(saveto/'CV_scores.csv')

    # All CV predictions
    all_res_df, res_df = make_pred_df(y, group, dose, _pred, _probas, labels)
    all_res_df.to_csv(saveto/'all_CV_results.csv')
    res_df.to_csv(saveto/'CV_results.csv')

    ## Plot confusion matrix
    #------------------------------
    _plot_confusion_matrix(
        res_df, title=None, savefig=saveto/'figure3_CV_confusion_matrix.svg')

    return rf_random.best_estimator_