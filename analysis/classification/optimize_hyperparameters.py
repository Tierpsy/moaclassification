#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Function to optimize the classifier hyperparameters for the moa classification
task.

Uses the RandomizedSearchCV algorithm of sklearn to select the optimal
parameters from a parameter grid using 4 fold cross validation. It searches
across N_ITER different random combinations of the parameters.

The hyperparameter selection is done based on classification accuracy
of individual replicates (not compound-level predictions).
Ideally, we would want to select the parameters based the compound-level
classification accuracy. However, selecting based on replicate-level
accuracy is accepted as an alternative because the two scores
are highly correlated, which means that in most cases the higest compound-level
accuracy is achieved when we have the highest replicate-level accuracy.

Created on Thu Jul 30 11:12:48 2020

@author: em812
"""

import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import clone
from tierpsytools.drug_screenings.MIL.majority_vote import \
    majority_vote_CV
from moaclassification.classification_helper import \
    make_score_df, make_pred_df, _plot_confusion_matrix

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
    pipeline : pipeline object with fit and predict methods.
        The pipeline object to optimize. It is expected to have an 'estimator'
        component.
    cv : int or sklearn splitter object
        See cv parameter of RandomizedSearchCV.
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
    pipeline object
        The pipeline object with the optimal parameters, trained using the
        entire dataset in feat.

    Saves
    -----
    - a dictionary with the optimal parameters for the estimator in the pipeline
    in best_params.p
    - the trained RandomizedSearchCV object in 'fitted_RandomizedSearchCV.p'
    - the CV results at replicate level obtained with the optimal parameters
    in all_CV_results.csv
    - the CV results at compound level obtained with the optimal parameters in
    CV_results.csv

    Plots
    -----
    - the confusion matrix with the CV predictions in figure 3B

    """

    saveto = saveroot / 'optimize_hyperparameters_results'
    saveto.mkdir(parents=True, exist_ok=True)

    ## Use the random grid to search for best hyperparameters
    # Random search of parameters, using 4 fold cross validation,
    # search across N_ITER different combinations.
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