#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 14:07:23 2020

@author: em812
"""
import numpy as np

def get_feat_set(feat_set, align_bluelight=False):
    from pathlib import Path
    import pandas as pd
    from moaclassification import AUX_FILES_DIR

    if not isinstance(feat_set, str):
        raise ValueError('feat_to_select must be a string')

    tierpsy_sets_root = Path(AUX_FILES_DIR)

    tierpsy_set_file = tierpsy_sets_root/ (feat_set+'.csv')

    cols = pd.read_csv(tierpsy_set_file, header=None)[0].to_list()

    if align_bluelight:
        bluelight = ['prestim', 'bluelight', 'poststim']
        cols = ['_'.join([col, blue]) for col in cols for blue in bluelight]

    return cols

def add_key(key, scores, scores_maj, scorenames):
    scores[key] = {score:[] for score in scorenames}
    scores_maj[key] = {score:[] for score in scorenames}
    return scores, scores_maj

def update_results(key, scores, scores_maj, _scores, _scores_maj):
    if _scores is not None:
        scorenames = list(_scores.keys())
    else:
        scorenames = list(_scores_maj.keys())
    for score in scorenames:
        if _scores is not None:
            scores[key][score] = _scores[score]
        else:
            scores[key][score] = np.ones(len(_scores_maj[score]))*np.nan
        if _scores_maj is not None:
            scores_maj[key][score] = _scores_maj[score]
        else:
            scores_maj[key][score] = np.ones(len(_scores[score]))*np.nan

    return scores, scores_maj

def append_to_key(key, scores, scores_maj, _scores, _scores_maj):
    scorenames = list(_scores.keys())
    for score in scorenames:
        scores[key][score].append(_scores[score])
        scores_maj[key][score].append(_scores_maj[score])
    return scores, scores_maj

def get_drug2moa_mapper(drug_id, moa_id):
    drug_id = np.array(drug_id)
    moa_id = np.array(moa_id)

    drugs, ind = np.unique(drug_id, return_index=True)
    moas = moa_id[ind]

    return dict(zip(drugs, moas))

def random_permutation(array_list):

    for x in array_list:
        assert len(array_list[0])==len(x)

    p = np.random.permutation(len(array_list[0]))

    return (x[p] for x in array_list)


def sort_arrays(array_list, by_array):

    ind = np.argsort(by_array)

    return (x[ind] for x in array_list)