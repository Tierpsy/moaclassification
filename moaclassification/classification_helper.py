#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 18:55:27 2020

@author: em812
"""

import pandas as pd
import matplotlib.pyplot as plt

def make_pred_df(y, drug_type, drug_dose, pred, probas, labels):
    from tierpsytools.drug_screenings.MIL.majority_vote import \
        get_majority_vote, _get_y_group

    # Make a dataframe with the predictions of the classifier (class labels and
    # and probabilities) for all the replicates
    all_res_df = pd.DataFrame({
        'y': y,
        'drug_type': drug_type,
        'drug_dose': drug_dose,
        'prediction': pred
        })
    all_res_df = pd.concat([
        all_res_df, pd.DataFrame(probas, columns=labels)], axis=1)

    # Majority vote:
    # Make a dataframe with the predictions of the classifier (class labels)
    # at the compound level
    y_maj = _get_y_group(y, drug_type)
    pred_maj = get_majority_vote(drug_type, y_pred=pred)
    drug_type_maj = y_maj.index.to_list()

    res_df = pd.DataFrame({
        'drug_type': drug_type_maj,
        'y': y_maj,
        'prediction': pred_maj.loc[drug_type_maj]
        })

    return all_res_df, res_df

def make_score_df(scores, scores_maj, key=None):

    if scores is not None:
        score_df = pd.concat([
            pd.DataFrame(
                {'cv_score': scores[score],
                 'sample_type': ['standard']*len(scores[score]),
                 'score': [score]*len(scores[score]),
                 'fold': list(range(len(scores[score])))
                 })
                for score in scores.keys()
                    ])
    else:
        score_df = pd.DataFrame()

    if scores_maj is not None:
        score_maj_df = pd.concat([
            pd.DataFrame(
                {'cv_score': scores_maj[score],
                 'sample_type': ['majority']*len(scores_maj[score]),
                 'score': [score]*len(scores_maj[score]),
                 'fold': list(range(len(scores_maj[score])))
                 })
                for score in scores_maj.keys()
                    ])
    else:
        score_maj_df = pd.DataFrame()

    score_df = pd.concat([score_df, score_maj_df], axis=0)

    if key is not None:
        score_df = score_df.assign(key=key)

    return score_df

def get_best_feat_size(score_df, select_based_on='accuracy'):
    """
    Get the best number of features based on the score_df that contains the
    CV results in the format created by the make_score_df function.
    """
    mask = score_df['score']==select_based_on
    mean_cv_score = score_df[mask].groupby(by=['key', 'sample_type']).mean()
    best_n_feat = mean_cv_score['cv_score'].idxmax()[0]
    return best_n_feat


def _plot_confusion_matrix(res_df, classes=None, title=None, savefig=None):
    """
    Plot a compound-level confusion matrix based on the res_df that contains
    results in the format created by the make_pred_df function.
    """
    from sklearn.metrics import accuracy_score
    from tierpsytools.analysis.classification_tools import plot_confusion_matrix

    if title is None:
        acc = accuracy_score(res_df['y'], res_df['prediction'])
        title = 'CV accuracy = {:.2f}'.format(acc)

    plot_confusion_matrix(
        res_df['y'], res_df['prediction'], classes=classes,
        title=title, figsize=(8,8), saveto=savefig)

    return

def _plot_cv_results_by_key(score_df, saveto=None):

    import seaborn as sns

    scorenames = score_df['score'].unique()

    for score in scorenames:
        fig, ax = plt.subplots(figsize=(10,10))
        sns.lineplot(data=score_df[score_df['score']==score], x='key',
                     y='cv_score', style='sample_type', ax=ax)
        ax.set_ylabel(score)
        rc_params = {
        'font.sans-serif': "Arial",  # just an example
        'svg.fonttype': 'none',
        }
        with plt.rc_context(rc_params):
            plt.savefig(saveto/'{}_CV_results.svg'.format(score))
    return
