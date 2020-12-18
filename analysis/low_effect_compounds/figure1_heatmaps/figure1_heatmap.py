#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find compounds with low effect among the entire databese of screened compounds.
Drop these compounds from the dataset and split remaining compoudns to
train / test set stratified by MOA groups.

Created on Thu Sep 17 21:17:52 2020

@author: em812
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from moaclassification import INPUT_DIR, ANALYSIS_DIR
plt.rcParams['svg.fonttype'] = 'none'

#%% Input
# Input files
root_res = Path(ANALYSIS_DIR) / 'low_effect_compounds' / 'results'

pval_file = root_res / 'pvals_LMM.csv'
low_eff_file = root_res / 'low_effect_drugs.csv'
signif_eff_file = root_res / 'signif_effect_drugs.csv'

moa_file = Path(INPUT_DIR) / 'AllCompoundsMoA.csv'

saveto = Path().cwd() / 'heatmaps'
saveto.mkdir(exist_ok=True)

#%% Read data
moa_info = pd.read_csv(moa_file)

moamapper = dict(moa_info[['CSN', 'MOA_group']].values)

pvals_df = pd.read_csv(pval_file, index_col=0)
low_effect_drugs = pd.read_csv(low_eff_file)['low_effect_drugs'].to_list()
signif_effect_drugs = pd.read_csv(signif_eff_file)['signif_effect_drugs'].to_list()

#%% Get number of significant features per bluelight condition for each compound
blue_conditions = ['prestim', 'bluelight', 'poststim']

# feature names for each bluelight condition
blue_columns = {
    blue: [col for col in pvals_df.columns if blue in col]
    for blue in blue_conditions}

# count significant features
n_sign = pd.concat([
    pd.Series((pvals_df[blue_columns[blue]]<0.01).sum(axis=1), name=blue)
    for blue in blue_conditions
    ], axis=1)

# get the class id (MOA_group) of each compound to sort them
n_sign = n_sign.assign(MOA_group=n_sign.index.map(moamapper))
n_sign = n_sign.sort_values(by=['MOA_group', 'prestim'])

## Find class with largest diffences
differences = n_sign.groupby(by='MOA_group').agg(lambda x: x.max(axis=0)-x.min(axis=0))

# get the clas name (MOA_general) and drug names for the plot labels
mapper = dict(moa_info[['MOA_group', 'MOA_general']].values)
n_sign = n_sign.assign(MOA_general = n_sign['MOA_group'].map(mapper))

mapper = dict(moa_info[['CSN', 'drug_name']].values)
n_sign = n_sign.assign(drug_name = n_sign.index.map(mapper))

# get n significant in logarithmic scale for plotting
n_sign[blue_conditions] = np.log10(n_sign[blue_conditions])
n_sign = n_sign.replace({-np.inf:0})

#%% Plot one heatmap with all compounds
# set font size
sns.set(font_scale=0.7)

# define colormap
cmap = sns.cubehelix_palette(start=.5, rot=-.75, dark=0.1, reverse=True, as_cmap=True)

# define parameters to save all labels as text in svgs
rc_params = {
    'font.sans-serif': "Arial",  # just an example
    'svg.fonttype': 'none',
    }

# plot heatmap
fig, ax = plt.subplots(figsize=(3,15))
g = sns.heatmap(n_sign[blue_conditions].set_index(n_sign['MOA_general']), ax=ax, cmap=cmap)

# Set ticks and tick labels
cbar = g.collections[0].colorbar
cbar.set_ticks([0, 1, 2, 3 ])
cbar.set_ticklabels(['0', '10', '100', '1000'])

# plot horizontal lines to separate classes
ax.hlines(n_sign['MOA_group'].value_counts(sort=False).cumsum().to_list(),
          *ax.get_xlim(), color='white')

# rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
plt.tight_layout()

# save figure
with plt.rc_context(rc_params):
    plt.savefig(saveto/'heatmap.svg')
    plt.savefig(saveto/'heatmap.pdf')

# %% plot separate heatmap per moa
for moa in n_sign.MOA_group.unique():
    mask = (n_sign['MOA_group']==moa).values
    fig, ax = plt.subplots()
    g = sns.heatmap(n_sign.set_index('drug_name').loc[mask, blue_conditions], ax=ax, cmap=cmap)
    cbar = g.collections[0].colorbar
    cbar.set_ticks([0, 1, 2, 3 ])
    cbar.set_ticklabels(['0', '10', '100', '1000'])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.tight_layout()
    with plt.rc_context(rc_params):
        plt.savefig(saveto/'heatmap_moa={}.pdf'.format(moa))
        plt.close()

