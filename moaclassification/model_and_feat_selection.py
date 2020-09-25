#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 16:12:10 2020

@author: em812
"""

def RFE_selection(X, y, n_feat, estimator, step=100, save_to=None):
    from sklearn.feature_selection import RFE
    from time import time
    import pickle

    print('RFE selection for n_feat={}.'.format(n_feat))
    start_time = time()

    rfe = RFE(estimator, n_features_to_select=n_feat, step=step)
    X_sel = rfe.fit_transform(X,y)

    print("RFE: --- %s seconds ---" % (time() - start_time))

    if save_to is not None:
        pickle.dump( rfe, open(save_to/'fitted_rfe_nfeat={}.p'.format(n_feat), "wb") )

    return X_sel, rfe.support_, rfe

def kbest_selection(X, y, n_feat, score_func=None):
    from sklearn.feature_selection import SelectKBest, f_classif

    if score_func is None:
        score_func = f_classif

    selector = SelectKBest(score_func=score_func, k=n_feat)
    X_sel = selector.fit_transform(X, y)

    return X_sel, selector.support_


def model_selection(
        X, y,
        estimator,
        param_grid,
        cv_strategy=0.2,
        save_to=None,
        saveid=None
        ):
    from sklearn.model_selection import GridSearchCV
    from time import time
    import pickle

    print('Starting grid search CV...')
    start_time = time()
    grid_search = GridSearchCV(
        estimator, param_grid=param_grid, cv=cv_strategy, n_jobs=-1, return_train_score=True)

    grid_search.fit(X, y)
    print("Grid search: --- %s seconds ---" % (time() - start_time))

    if save_to is not None:
        pickle.dump( grid_search, open(save_to/'fitted_gridsearchcv_nfeat={}.p'.format(saveid), "wb") )

    return grid_search.best_estimator_, grid_search.best_score_

def plot_cv_accuracy(n_feat_vector, cv_accuracy, n_moas, savefig):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.title('{} MOAs - max cv accuracy = {}'.format(n_moas, max(cv_accuracy)))
    # plt.plot(n_feat_vector, nfeat_gs_accuracy, label='grid search accuracy')
    plt.plot(n_feat_vector, cv_accuracy, label='cv accuracy')
    plt.ylim([0,1.0])
    plt.xlabel('n features')
    plt.ylabel('MOA classification accuracy')
    plt.legend()
    plt.xscale('log')
    plt.savefig(savefig)
    plt.close()

    return