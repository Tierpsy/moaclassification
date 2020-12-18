#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 13:02:49 2020

@author: em812
"""
import pandas as pd
import numpy as np

def get_novelty_score(svms, thetas, svm_feats):
    # Get the known/novel predictions from all estimators of the ensemble
    predictions = {}
    for moa in svms.keys():
        # Get estimator and weight
        weight = svms[moa]['weight']
        est = svms[moa]['estimator']

        # Exclude the points that where predicted to belong to the presumed-novel
        # class of this svm's partition (according to [1])
        mask = thetas[moa]['most_likely_class']!=moa

        predictions[moa] = est.predict(thetas[moa].loc[mask, svm_feats])
        predictions[moa] = pd.Series(
            predictions[moa]*weight, index=thetas[moa][mask].index)

    # Get novelty score for each compound: weighted average of predictions
    predictions = pd.concat(predictions, axis=1)
    sum_weights = np.sum([w for e,w in svms.values()])
    novelty_score = predictions.apply( lambda x: np.nansum(x)/sum_weights, axis=1 ) #

    return novelty_score

def theta_score(probas, group, labels):

    # mean probas per in drug
    probas_mean = pd.DataFrame(probas).groupby(by=group).mean()

    # get two most likely
    most_likely_ids = np.flip(np.argsort(probas_mean.values, axis=1), axis=1)[:, :2]
    theta = {}
    most_likely_class = {}
    for i, grp in enumerate(probas_mean.index):
        theta[grp] = probas_mean.values[i, most_likely_ids[i, 0]] / probas_mean.values[i, most_likely_ids[i, 1]]
        most_likely_class[grp] = labels[most_likely_ids[i, 0]]

    return pd.Series(theta, name='theta_score'), pd.Series(most_likely_class, name='most_likely_class')

def get_theta_i_per_class(theta_in, moas=None, groups=None, group2moa_mapper=None):

    if moas is None:
        if groups is None or group2moa_mapper is None:
            raise ValueError('Must define either moas or (groups and mapper).')

    if moas is None:
        moas = pd.Series(groups).map(group2moa_mapper)

    theta_i_mean = theta_in.groupby(by=moas).mean()

    return theta_i_mean

def collect_theta_info(theta_s, o_s, theta_i, moas=None):

    # Make dataframe with theta scores and moa or most_likely moa
    theta_s = pd.concat([theta_s, o_s], axis=1)
    if moas is not None:
        theta_s = pd.concat([theta_s, moas], axis=1)
        theta_s = theta_s.assign(is_correct_class=(o_s==moas))
    # Add the mean theta for the associated moa
    if moas is None:
        theta_s = theta_s.assign(moa_theta_i=o_s.map(theta_i))
    else:
        theta_s = theta_s.assign(moa_theta_i=moas.map(theta_i))
    # Add the ratio
    theta_s = theta_s.assign(ratio=theta_s['theta_score']/theta_s['moa_theta_i'])

    return theta_s

def get_theta_scores_in(probas, group, labels, group2moa_mapper=None):
    # theta scores for drugs
    theta_s, o_s = theta_score(probas, group, labels)

    # theta_i : average theta score per class i
    moas = pd.Series(theta_s.index).map(group2moa_mapper).values
    theta_i = get_theta_i_per_class(theta_s, moas=moas)

    theta_s = collect_theta_info(
        theta_s, o_s, theta_i, moas=pd.Series(moas, name='true_MOA_group', index=theta_s.index))

    return theta_s, theta_i

def get_theta_scores_out(probas, group, labels, theta_i):
    # theta scores for drugs
    theta_s, o_s = theta_score(probas, group, labels)

    theta_s = collect_theta_info(theta_s, o_s, theta_i)

    return theta_s

