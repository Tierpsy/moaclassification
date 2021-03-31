#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 10:30:13 2021

@author: em812
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from moaclassification import ANALYSIS_DIR

# Input
root_res = Path(ANALYSIS_DIR) / 'low_effect_compounds' / 'results'
pval_file = root_res / 'null_pvals_LMM.csv'

# define parameters to save all labels as text in svgs
rc_params = {
    'font.sans-serif': "Arial",  # just an example
    'svg.fonttype': 'none',
    }

saveto = Path().cwd() / 'heatmaps'
saveto.mkdir(exist_ok=True)

#%% Read data
pvals_df = pd.read_csv(pval_file, index_col=0)

#%%
blue_conditions = ['prestim', 'bluelight', 'poststim']
# feature names for each bluelight condition
blue_columns = {
    blue: [col for col in pvals_df.columns if blue in col]
    for blue in blue_conditions}

n_sign = pd.concat([
    pd.Series((pvals_df[blue_columns[blue]]<0.01).sum(axis=1), name=blue)
    for blue in blue_conditions
    ], axis=1)

n_sign.index = ['DMSO']
cmap = sns.cubehelix_palette(start=.5, rot=-.75, dark=0.1, reverse=True, as_cmap=True)

fig, ax = plt.subplots(figsize=(7,2))
g = sns.heatmap(n_sign[blue_conditions], ax=ax, cmap=cmap)
cbar = g.collections[0].colorbar
cbar.set_ticks([0, 1, 2, 3 ])
cbar.set_ticklabels(['0', '10', '100', '1000'])
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()
with plt.rc_context(rc_params):
    plt.savefig(saveto/'heatmap_null.pdf')
    plt.close()

