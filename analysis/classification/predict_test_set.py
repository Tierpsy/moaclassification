#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:40:39 2020

@author: em812
"""
from tierpsytools.analysis.helper import _get_multi_sclassifscorers
from moaclassification.classification_helper import \
    make_score_df, make_pred_df, _plot_confusion_matrix

def main_predict_test_set(
        feat_test, y_test, group_test, dose_test,
        trained_estimator, cv, class_labels,
        vote_type, scorer, scorenames,
        saveroot):

    scorers = _get_multi_sclassifscorers(scorer)

    saveto = saveroot / 'predict_test_set_results'
    saveto.mkdir(parents=True, exist_ok=True)

    #%% Predict
    pred_test = trained_estimator.predict(feat_test)
    probas_test = trained_estimator.predict_proba(feat_test)

    # Get all the scores
    scores_test = {score:[] for score in scorenames}
    scores_maj_test = {score:[] for score in scorenames}
    for key in scorers.keys():
        scores_test[key].append(scorers[key].score(
            y_test, pred=pred_test, probas=probas_test, labels=class_labels
            ))
        scores_maj_test[key].append(scorers[key].score_maj(
            y_test, group_test, pred=pred_test, probas=probas_test,
            labels=class_labels, vote_type=vote_type
            ))

    #%% Store scores and predictions
    # Test scores
    score_df = make_score_df(scores_test, scores_maj_test)
    score_df.to_csv(saveto/'test_scores.csv')

    # Test predictions
    all_res_df, res_df = make_pred_df(
        y_test, group_test, dose_test, pred_test, probas_test, class_labels)
    all_res_df.to_csv(saveto/'all_test_results.csv')
    res_df.to_csv(saveto/'test_results.csv')

    #%% Plot confusion matrix
    _plot_confusion_matrix(
        res_df, title=None, savefig=saveto/'figure3_test_set_confusion_matrix.svg')

    return