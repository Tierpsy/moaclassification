#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 10:16:17 2020

@author: em812
"""
import pdb
from tierpsytools.analysis.drug_screenings.MIL.effect_size_transform \
    import effectSizeTransform
from tierpsytools.analysis.drug_screenings.drug_class import drugClass
from tierpsytools.feature_processing.scaling_class import scalingClass
from tierpsytools.analysis.classification_tools import get_fscore, \
    plot_confusion_matrix
import model_and_feat_selection as tsel
# import pdb
import numpy as np
import pickle
from time import time

def effect_size_parameters_one_strain(
        inparam, n_feat_vector, feat, meta,
        estimator, param_grid,
        rfe_step, cv_strategy,
        save_models_to=None, save_to=None,
        feat_test=None, meta_test=None
        ):
    """


    Parameters
    ----------
    inparam : TYPE
        DESCRIPTION.
    n_feat_vector : TYPE
        DESCRIPTION.
    feat : TYPE
        DESCRIPTION.
    meta : TYPE
        DESCRIPTION.
    estimator : TYPE
        DESCRIPTION.
    param_grid : TYPE
        DESCRIPTION.
    rfe_step : TYPE
        DESCRIPTION.
    cv_strategy : TYPE
        DESCRIPTION.
    save_models_to : TYPE, optional
        DESCRIPTION. The default is None.
    save_to : TYPE, optional
        DESCRIPTION. The default is None.
    feat_test : TYPE, optional
        DESCRIPTION. The default is None.
    meta_test : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """

    res = {}
    res['params'] = inparam
    res['n_feat_vector'] = n_feat_vector
    if not inparam['scale_per_compound']:
        scaler_samples = scalingClass(function = inparam['scale_samples'])
        feat = scaler_samples.fit_transform(feat)

    #%% Effect size parameters
    # Create drug instances
    print('Creating drug instances...'); st_time=time()
    namelist = meta.loc[meta['drug_type']!='DMSO', 'drug_type'].unique()

    druglist = [drugClass(
        feat[meta['drug_type']==idg], meta[meta['drug_type']==idg], MOAinfo=True)
        for idg in namelist]
    print('Done in {:.0f} sec.'.format( time()-st_time ))

    control = feat[meta['drug_type']=='DMSO']

    print('Effect size parameters transformation...'); st_time=time()
    est = effectSizeTransform(**inparam)
    est.fit(druglist, control)
    print('Done in {:.0f} sec.'.format( time()-st_time ))

    est.parameters_mask = (np.std(est.effect_size_parameters, axis=0) != 0) & \
        (~np.isnan(np.std(est.effect_size_parameters, axis=0) ))
    parameters = est.scale_parameters(apply_mask=True)
    if parameters.shape[1]==0:
        res['error'] = 'empty parameters matrix'
        return res

    y = np.array([drug.moa_group for drug in druglist])

    #%% Feature and model selection
    print('Starting feature and model selection...'); st_time=time()
    cv_accuracy = []
    selectors = []
    classifiers = []
    for n_feat in n_feat_vector:
        # Feature selection
        param, _, selector = tsel.RFE_selection(
            parameters, y, n_feat, estimator, step=rfe_step, save_to=save_models_to)

        # Model selection
        estimator, cv_acc = tsel.model_selection(
            param, y, estimator, param_grid, cv_strategy=cv_strategy, save_to=save_models_to, saveid=n_feat)

        # Get mean accuracy from different folds
        selectors.append(selector)
        classifiers.append(estimator)
        cv_accuracy.append(cv_acc)
    print('Done in {:.0f} sec.'.format( time()-st_time ))
    # Plot results
    if len(n_feat_vector)>1:
        savefig = save_to/'cv_accuracy_vs_nfeat.png'
        n_feat_vector = [x if (x is not None) else druglist[0].feat.shape[1] for x in n_feat_vector]
        tsel.plot_cv_accuracy(n_feat_vector, cv_accuracy, np.unique(y).shape[0], savefig=savefig)

    res['cv_acc'] = cv_accuracy

    # Get max CV accuracy and select best feature set and model hyperparameters
    best_id = np.argmax(cv_accuracy)

    selector = selectors[best_id]
    estimator = classifiers[best_id]
    del selectors, classifiers

    #%% Test set
    if feat_test is None:
        return res

    # Preprocessing
    if not inparam['scale_per_compound']:
        feat_test = scaler_samples.transform(feat_test)

    ## Effect size parameter transform
    namelist_test = meta_test['drug_type'].unique()
    druglist_test = [drugClass(
        feat_test[meta_test['drug_type']==idg],
        meta_test[meta_test['drug_type']==idg], MOAinfo=True)
        for idg in namelist_test]

    est_test = effectSizeTransform(**inparam)

    parameters_test = est_test.fit_transform(druglist_test, control)
    parameters_test = est.param_scaler.transform(parameters_test[:, est.parameters_mask])

    y_test = np.array([drug.moa_group for drug in druglist_test])
    res['y_real'] = y_test

    ## Classification
    estimator.fit(parameters[:, selector.support_], y)
    try:
        ypred = estimator.predict(parameters_test[:, selector.support_])
        test_acc = estimator.score(parameters_test[:, selector.support_], y_test)
    except:
        res['y_pred'] = None
        res['test_acc'] = None
        res['error'] = 'nans in test parameters.'
        return res

    res['fscore'] = get_fscore(y_test, ypred)

    classes={drug.moa_group:drug.moa_label for drug in druglist}
    plot_confusion_matrix(
        y_test, ypred, classes=classes, saveto=save_to/'confusion_matrix.png',
        title='Test accuracy = {:.3f}'.format(test_acc))

    res['y_pred'] = ypred
    res['test_acc'] = test_acc

    if save_to is not None:
        pickle.dump(res, open(save_to/'res.p', "wb"))

    return res

#%%
def effect_size_parameters_two_strains(
        inparam, n_feat_vector, features, metadata,
        estimator, param_grid,
        rfe_step, cv_strategy,
        save_models_to=None, save_to=None,
        features_test=None, metadata_test=None
        ):
    """


    Parameters
    ----------
    inparam : TYPE
        DESCRIPTION.
    n_feat_vector : TYPE
        DESCRIPTION.
    features : TYPE
        DESCRIPTION.
    metadata : TYPE
        DESCRIPTION.
    estimator : TYPE
        DESCRIPTION.
    param_grid : TYPE
        DESCRIPTION.
    rfe_step : TYPE
        DESCRIPTION.
    cv_strategy : TYPE
        DESCRIPTION.
    save_models_to : TYPE, optional
        DESCRIPTION. The default is None.
    save_to : TYPE, optional
        DESCRIPTION. The default is None.
    features_test : TYPE, optional
        DESCRIPTION. The default is None.
    metadata_test : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    res : TYPE
        DESCRIPTION.

    """

    res = {}
    res['params'] = inparam
    res['n_feat_vector'] = n_feat_vector
    if not inparam['scale_per_compound']:
        scaler_samples = []
        for i in range(len(features)):
            scaler_samples.append(scalingClass(function = inparam['scale_samples']))
            features[i] = scaler_samples.fit_transform(features[i])

    #%% Effect size parameters
    namelist = [meta.loc[meta['drug_type']!='DMSO', 'drug_type'].unique()
                for meta in metadata]
    namelist = list(set(namelist[0]).union(set(namelist[1])))

    parameters = []
    masks = []
    y = []
    for feat, meta in zip(features, metadata):
        # Create drug instances
        print('Creating drug instances...'); st_time=time()
        druglist = [drugClass(
            feat[meta['drug_type']==idg], meta[meta['drug_type']==idg], MOAinfo=True)
            for idg in namelist]
        print('Done in {:.0f} sec.'.format( time()-st_time ))

        control = feat[meta['drug_type']=='DMSO']

        print('Effect size parameters transformation...'); st_time=time()
        est = effectSizeTransform(**inparam)
        est.fit(druglist, control)
        print('Done in {:.0f} sec.'.format( time()-st_time ))

        est.parameters_mask = (np.std(est.effect_size_parameters, axis=0) != 0) & \
            (~np.isnan(np.std(est.effect_size_parameters, axis=0) ))

        parameters.append(est.effect_size_parameters[:, est.parameters_mask])
        masks.append(est.parameters_mask)
        y.append( np.array([drug.moa_group for drug in druglist]) )

        if parameters[-1].shape[1]==0:
            res['error'] = 'empty parameters matrix'
            return res

    del est
    assert np.all(y[0] == y[1])
    y = y[0]
    parameters = np.concatenate(parameters, axis=1)
    scaler_param = scalingClass(function = inparam['scale_params'])
    parameters = scaler_param.fit_transform(parameters)


    #%% Feature and model selection
    print('Starting feature and model selection...'); st_time=time()
    cv_accuracy = []
    selectors = []
    classifiers = []
    for n_feat in n_feat_vector:
        # Feature selection
        param, _, selector = tsel.RFE_selection(
            parameters, y, n_feat, estimator, step=rfe_step, save_to=save_models_to)

        # Model selection
        estimator, cv_acc = tsel.model_selection(
            param, y, estimator, param_grid, cv_strategy=cv_strategy, save_to=save_models_to, saveid=n_feat)

        # Get mean accuracy from different folds
        selectors.append(selector)
        classifiers.append(estimator)
        cv_accuracy.append(cv_acc)
    print('Done in {:.0f} sec.'.format( time()-st_time ))
    # Plot results
    if len(n_feat_vector)>1:
        savefig = save_to/'cv_accuracy_vs_nfeat.png'
        n_feat_vector = [x if (x is not None) else druglist[0].feat.shape[1] for x in n_feat_vector]
        tsel.plot_cv_accuracy(n_feat_vector, cv_accuracy, np.unique(y).shape[0], savefig=savefig)

    res['cv_acc'] = cv_accuracy

    # Get max CV accuracy and select best feature set and model hyperparameters
    best_id = np.argmax(cv_accuracy)

    selector = selectors[best_id]
    estimator = classifiers[best_id]
    del selectors, classifiers

    #%% Test set
    if features_test is None:
        return res

    # Preprocessing
    if not inparam['scale_per_compound']:
        for i in range(len(features_test)):
            features_test[i] = scaler_samples[i].transform(features_test[i])

    ## Effect size parameter transform
    namelist_test = [meta['drug_type'].unique()
                for meta in metadata_test]
    namelist_test = list(set(namelist_test[0]).union(set(namelist_test[1])))

    parameters_test = []
    y_test = []
    for feat_test, meta_test in zip(features_test, metadata_test):
        druglist_test = [drugClass(
            feat_test[meta_test['drug_type']==idg],
            meta_test[meta_test['drug_type']==idg], MOAinfo=True)
            for idg in namelist_test]

        est_test = effectSizeTransform(**inparam)

        est_test.fit(druglist_test, control)
        parameters_test.append( est_test.effect_size_parameters )

        y_test.append( np.array([drug.moa_group for drug in druglist_test]) )

    assert np.all(y_test[0]==y_test[1])
    y_test = y_test[0]
    res['y_real'] = y_test

    parameters_test = np.concatenate(
        [param[:, mask] for param,mask in zip(parameters_test, masks)], axis=1)


    ## Classification
    estimator.fit(parameters[:, selector.support_], y)
    try:
        ypred = estimator.predict(parameters_test[:, selector.support_])
        test_acc = estimator.score(parameters_test[:, selector.support_], y_test)
    except:
        res['y_pred'] = None
        res['test_acc'] = None
        res['error'] = 'nans in test parameters.'
        return res

    res['fscore'] = tsel.get_fscore(y_test, ypred)

    classes={drug.moa_group:drug.moa_label for drug in druglist}
    plot_confusion_matrix(
        y_test, ypred, classes=classes, saveto=save_to/'confusion_matrix.png',
        title='Test accuracy = {:.3f}'.format(test_acc))

    res['y_pred'] = ypred
    res['test_acc'] = test_acc

    if save_to is not None:
        pickle.dump(res, open(save_to/'res.p', "wb"))

    return res