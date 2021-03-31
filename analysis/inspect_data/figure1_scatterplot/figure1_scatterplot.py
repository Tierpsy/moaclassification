#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:26:20 2020

@author: em812
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from moaclassification import INPUT_DIR

#%% Input
# Paths to data
data_file = Path(INPUT_DIR) / 'long_format'/ 'features_long_format.csv'
meta_file = Path(INPUT_DIR) / 'long_format'/ 'metadata_long_format.csv'
moa_file = Path(INPUT_DIR) / 'AllCompoundsMoA.csv'

choose_drugs_file = 'chosen_drugs_doses.csv'

saveto = Path().cwd() / 'scatterplot'
saveto.mkdir(exist_ok=True)

#%% Read and preprocess date
feat = pd.read_csv(data_file, index_col=None)
meta = pd.read_csv(meta_file, index_col=None)

moa = pd.read_csv(moa_file)
choose_drugs = pd.read_csv(choose_drugs_file)

#%% Clean up data
#----------------------------
# Choose only prestimulus
ids = (meta['bluelight']=='prestim')
feat = feat[ids]
meta = meta[ids]

# Separate the chosen MOAs to highlight
ids = meta['MOA_group'].isin([0, 2, 7, 17])
featp = feat[ids]
metap = meta.loc[ids, ['drug_type', 'drug_dose']]

# Keep only chosen drugs in the separate dfs
ids = metap['drug_type'].isin(choose_drugs['drug_type'].to_list()+['DMSO'])
metap = metap[ids]
featp = featp[ids]

# Only keep the chosen doses in the separate dfs
for drug, dose in choose_drugs[['drug_type', 'drug_dose']].values:
    ids = (metap['drug_type']==drug) & (metap['drug_dose']!=dose)
    featp = featp[~ids]
    metap = metap[~ids]

moamapper =  dict(moa[['CSN', 'MOA_general']].values)

#%% Plot scatterplot
ft_to_plot = ['curvature_mean_tail_abs_90th', 'speed_90th']
error = 'std'

# First plot average points of all drugs at all doses (background grey points)
# without errorbars
data=pd.concat([feat[ft_to_plot], meta[['drug_type', 'drug_dose']]], axis=1)
data = data.groupby(by=['drug_type', 'drug_dose']).agg(['mean', error])

plt.figure()
plt.rcParams.update({'legend.fontsize': 15,
         'axes.labelsize': 15,
         'axes.titlesize': 15,
         'xtick.labelsize': 13,
         'ytick.labelsize': 13,
         'svg.fonttype' : 'none',
         'font.sans-serif' : 'Arial'})

plt.scatter(
    x=data[ft_to_plot[0]]['mean'], y=data[ft_to_plot[1]]['mean'],
    marker='o', edgecolors='none', label='__nolegend__', color='grey',
    alpha=0.2)

plt.errorbar(
        x=data[ft_to_plot[0]]['mean'],
        y=data[ft_to_plot[1]]['mean'],
        xerr=data[ft_to_plot[0]][error],
        yerr=data[ft_to_plot[1]][error],
        marker=None, color='grey',
        alpha=0.2, linestyle='', elinewidth=0.5
        )

# Then plot the chosen drugs at the chosen doses
data=pd.concat([featp[ft_to_plot], metap[['drug_type', 'drug_dose']]], axis=1)

data = data.groupby(by='drug_type').agg(['mean', error])

for drug in data.index:
    plt.errorbar(
        x=data.loc[drug][ft_to_plot[0]]['mean'],
        y=data.loc[drug][ft_to_plot[1]]['mean'],
        xerr=data.loc[drug]['curvature_mean_tail_abs_90th'][error],
        yerr=data.loc[drug]['speed_90th'][error],
        marker='o', linestyle='', label=moamapper[drug]
        )

plt.legend()
plt.xlabel('tail curvature ($\mu m^{-1}$)', )
plt.ylabel('speed ($\mu m/s$)')
plt.xlim([0,0.024])
plt.tight_layout()
plt.savefig(saveto/'errorbar_{}.pdf'.format(error))
plt.savefig(saveto/'errorbar_{}.svg'.format(error))