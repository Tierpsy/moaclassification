#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:22:32 2020

@author: em812
"""
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from moaclassification import ANALYSIS_DIR

#%% Input
res_root = Path(ANALYSIS_DIR) / 'novelty_detection' / 'results'
novel_scores_file = res_root / 'novelty_scores_novel.csv'
test_scores_file = res_root / 'novelty_scores_test.csv'

#%% Read data
novel_scores = pd.read_csv(novel_scores_file, index_col=0)
test_scores = pd.read_csv(test_scores_file, index_col=0)

#%% Reformat data
test_scores = test_scores.assign(
    correct=test_scores[['y_true', 'y_pred']].apply(lambda x: x[0]==x[1], axis=1))

data = pd.concat([
    novel_scores.assign(compound_group='novel'),
    test_scores.assign(compound_group='not novel')
    ])

#%% Plot
sns.set(font_scale = 1.4, style='ticks')

rc_params = {
    'font.sans-serif': "Arial",  # just an example
    'svg.fonttype': 'none',
    }

fig,ax = plt.subplots()
g1 = sns.swarmplot(data=data, x='compound_group', y='novelty_score', ax=ax, size=10)
g1 = sns.swarmplot(data=data, x='compound_group', y='novelty_score', hue='correct', ax=ax, size=10)

ax.get_legend().remove()

g1.axes.set_xlabel('')
g1.axes.set_ylabel('novelty score')

with plt.rc_context(rc_params):
    plt.savefig('novelty_scores.pdf')
    plt.savefig('novelty_scores.svg')
