#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature selection function.
It uses an approach similar to the RFECV function of sklearn to select
the optimal feature set for the moa classification task. The differences from
the RFECV algorithm are the following:
    1) we pre-select the number of features to test in each iteration
    (instead of having a step that defined how many features to drop in each
     iteration)
    and
    2) we use the compound level prediction accuracy (majority vote score)
    as the criterion to select the best number of features

Created on Thu Jul 30 11:12:48 2020

@author: em812
"""

import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from tierpsytools.analysis.drug_screenings.MIL.majority_vote import \
    majority_vote_CV
from moaclassification.classification_helper import \
    make_score_df, make_pred_df, get_best_feat_size, _plot_cv_results_by_key
import pdb

#%% Global variables
# Selected feature set sizes to check
N_FEAT_TO_TEST = [2**i for i in range(10,6,-1)]

# How many steps in RFE to reach the target number of features
N_STEPS_PER_ITER = 10

# Which score to use to select the best feature set
SELECT_BASED_ON = 'accuracy'

#%% Main
def main_feature_selection(
        feat, y, group, dose, scaler, estimator, cv, vote_type,
        scorer, scorenames, saveroot
        ):
    """
    Parameters
    ----------
    feat : dataframe. shape=(n_samples, n_features)
        The dataframe containing the tierpsy features for each averaged
        bootstrapped datapoint.
    y : array-like, shape=(n_samples,)
        Class labels (moa id).
    group : array-like, shape=(n_samples,)
        Drug name for each sample.
    dose : array-like, shape=(n_samples,)
        Drug dose for each sample.
    scaler : scaler object with fit and fit_transform method
    estimator : classifier object with fit and predict method
    cv : int or splitter object with split method
    vote_type : 'counts' or 'probas'
        Defines the majority vote type.
    scorer : list of strings or scorer objects
        Defines the cv scores to estimate with the optimal parameter set.
    scorenames : list of strings
        The names of the scorers in the scorer parameter.
    saveroot : path
        Path to save the results.

    Returns
    -------
    list
        Best feature set for the moa classification task.

    Saves
    -----
    - All selected feature sets with the number of features defined in N_FEAT_TO_TEST
    - The CV scores (replicate level and compound level) for every selected
    feature set size in CV_scores.csv
    - The CV predictions at replicate level obtained with every selected
    feature set in all_CV_results_n_feat={*}.csv
    - The CV predictions at compound level obtained with every selected
    feature set in CV_results_n_feat={*}.csv

    Plots
    -----
    The CV scores for each N_FEAT_TO_TEST
    """

    saveto = saveroot / 'feature_selection_results'
    saveto.mkdir(parents=True, exist_ok=True)

    assert SELECT_BASED_ON in scorenames

    ## Initialize results dict
    score_df = []
    res_dfs = {}
    all_res_dfs = {}

    ##
    pipeline = Pipeline(
        [('scaler', scaler), ('estimator', estimator)]
        )

    ## All features
    #----------------------------
    n_feat = feat.shape[1]
    print('Predicting with: {} features'.format(n_feat))

    _score, _score_maj, _pred, _probas, labels =  majority_vote_CV(
        feat, y, group, clone(pipeline), cv, vote_type=vote_type,
        n_jobs=-1, scorer=scorer, return_predictions=True)

    score_df.append( make_score_df(_score, _score_maj, key=n_feat) )
    all_res_dfs[n_feat], res_dfs[n_feat] = make_pred_df(
        y, group, dose, _pred, _probas, labels)

    ## Feature selection
    #------------------------------
    # Begin with all features
    feat_set = feat.columns.to_numpy()

    # Loop over feature set sizes
    for n_feat in N_FEAT_TO_TEST:
        print('Predicting with: {} features'.format(n_feat))

        # Defined pipeline that scales, selects n_feat and fits classifier
        # without using any data from the cv-test set
        rfe = RFE(estimator=estimator, n_features_to_select=n_feat, step=100)
        rfe_pipe = Pipeline([
            ('scaler', scaler), ('selector', rfe), ('estimator', estimator)
            ])

        # Cross validation: get majority vote accuracy using the rfe pipeline
        _score, _score_maj, _pred, _probas, _ =  majority_vote_CV(
            feat, y, group, clone(rfe_pipe), cv, vote_type=vote_type,
            n_jobs=-1, scorer=scorer, return_predictions=True)

        # Append results
        score_df.append( make_score_df(_score, _score_maj, key=n_feat) )
        all_res_dfs[n_feat], res_dfs[n_feat] = make_pred_df(
            y, group, dose, _pred, _probas, labels)

        # Select n_features using the entire cv dataset
        step = int((feat_set.shape[0] - n_feat)/N_STEPS_PER_ITER)
        rfe = RFE(estimator=estimator, n_features_to_select=n_feat, step=step)

        rfe.fit(scaler.fit_transform(feat[feat_set]), y)
        feat_set = feat_set[rfe.support_]

        # Store selected feature set
        pd.Series(feat_set).to_csv(
            saveto/'selected_features_{}.csv'.format(n_feat),
            index=None, header=None)

    score_df = pd.concat(score_df)

    ## Store results
    #------------------------------
    # All CV scores
    score_df.to_csv(saveto/'CV_scores.csv')

    # All predictions
    for key in res_dfs.keys():
        all_res_dfs[key].to_csv(
            saveto/'all_CV_results_n_feat={}.csv'.format(key))
        res_dfs[key].to_csv(saveto/'CV_results_n_feat={}.csv'.format(key))

    ## Get best feature set
    #------------------------------
    best_n_feat = get_best_feat_size(score_df, SELECT_BASED_ON)
    best_feat_set = pd.read_csv(
        saveto/'selected_features_{}.csv'.format(best_n_feat),
        index_col=None, header=None)
    best_feat_set.to_csv(saveto/'best_feature_set.csv', index=None, header=None)

    ## Plot results
    #------------------------------
    _plot_cv_results_by_key(score_df, saveto=saveto)

    return best_feat_set[0].to_list()

