#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to make moa predictions for the test set with the optimal trained
estimator.

Created on Fri Nov 27 19:40:39 2020

@author: em812
"""
from tierpsytools.analysis.helper import _get_multi_sclassifscorers
from moaclassification.classification_helper import \
    make_score_df, make_pred_df, _plot_confusion_matrix

def main_predict_test_set(
        feat_test, y_test, group_test, dose_test,
        trained_estimator, cv,
        vote_type, scorer, scorenames,
        saveroot):
    """
    Parameters
    ----------
    feat_test : dataframe. shape=(n_samples, n_features)
        The dataframe containing the tierpsy features for each averaged
        bootstrapped datapoint of the test set.
    y_test : array-like, shape=(n_samples,)
        Class labels (moa id) for each sample in feat_test.
    group_test : array-like, shape=(n_samples,)
        Drug name for each sample in feat_test.
    dose_test : array-like, shape=(n_samples,)
        Drug dose for each sample in feat_test.
    trained_estimator : fitted estimator or pipeline object
        The optimal estimator trained using the entire CV set.
    cv : int or splitter object with split method.
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
    None.

    Saves
    -----
    - Test scores in CV_scores.csv
    - Test predictions at replicate level in all_test_results.csv
    - Test predictions at compound level in test_results.csv

    Plots
    -----
    - Confusion matrix of test predictions in Figure 3C

    """

    scorers = _get_multi_sclassifscorers(scorer)

    saveto = saveroot / 'predict_test_set_results'
    saveto.mkdir(parents=True, exist_ok=True)

    if hasattr(trained_estimator, 'classes_'):
        class_labels = trained_estimator.classes_
    else:
        class_labels = trained_estimator['estimator'].classes_

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
    test_acc = score_df.loc[
        (score_df['sample_type']=='majority') &
        (score_df['score']=='accuracy'), 'cv_score'].values[0]
    _plot_confusion_matrix(
        res_df, title='Test accuracy = {:.3f}'.format(test_acc), savefig=saveto/'figure3_test_set_confusion_matrix.svg')

    return