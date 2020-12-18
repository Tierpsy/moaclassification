#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 15:25:33 2020

@author: em812
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from moaclassification import INPUT_DIR
import pdb

DRUGS_TO_PLOT = {
    3: ['SY0376', 'SY0354', 'SY0353'],
    6: ['SY0622', 'SY0641', 'SY0612'],
    10: ['SY1048', 'SY1081', 'SY1021'],
    11: ['SY1154', 'SY1140', 'SY1124'],
    17: ['SY1793', 'SY1786', 'SY1713']
    }


#%% Input
# Paths to data
data_file = Path(INPUT_DIR) / 'long_format'/ 'features_long_format.csv'
meta_file = Path(INPUT_DIR) / 'long_format'/ 'metadata_long_format.csv'
moa_file = Path(INPUT_DIR) / 'AllCompoundsMoA.csv'

# split directory
train_file = Path(INPUT_DIR) / 'split' / 'train_compounds.csv'
test_file = Path(INPUT_DIR) / 'split' / 'test_compounds.csv'

saveroot = Path().cwd() / 'dose_response_curves'
saveroot.mkdir(exist_ok=True)

#%% Read and preprocess date
feat = pd.read_csv(data_file, index_col=None)
meta = pd.read_csv(meta_file, index_col=None)
moa = pd.read_csv(moa_file)

train_compounds = pd.read_csv(train_file)
test_compounds = pd.read_csv(test_file)

# Keep only train test compounds
meta = meta[meta['drug_type'].isin(
    train_compounds['drug_type'].to_list()+
    test_compounds['drug_type'].to_list()
    )]
feat = feat.loc[meta.index]

# Add the market names to the metadata
namemapper = dict(moa[['CSN', 'drug_name']].values)
meta['drug_name'] = meta['drug_type'].map(namemapper)

# Choose only prestimulus
ids = (meta['bluelight']=='prestim')
feat = feat[ids]
meta = meta[ids]


#%% Keep only some groups of features that are more interpretable
#----------------------------
fts = [col for col in feat.columns if any([s in col for s in ['curvature', 'speed', 'velocity', 'width']])]
fts = [col for col in fts if all([s not in col for s in ['path', 'blob']])]
fts = [col for col in fts if not col.startswith('d_')]

feat = feat[fts]

#%% Plot the most significant features of 3 drugs per moa
for moa, drugs in DRUGS_TO_PLOT.items():
    savefig = saveroot/'final_MOA={}'.format(moa)
    savefig.mkdir(exist_ok=True)
    for child in savefig.glob('*'):
        if child.is_file():
            child.unlink()

    ids = meta['drug_type'].isin(drugs+['DMSO'])

    selector = SelectKBest(k=20)
    # Impute nan values for the selection, because SelectKBest doesn't accept nans
    selector.fit(feat.fillna(feat.mean()).loc[ids], meta.loc[ids, 'drug_type'])
    # Get selected features for this drug
    fts_to_plot = feat.loc[:, selector.get_support()].columns
    pd.Series(fts_to_plot).to_csv(savefig/'plotted_features.csv')

    data=pd.concat([feat.loc[ids, fts_to_plot], meta.loc[ids]], axis=1)

    # Plot
    for ft in fts_to_plot:
        fig, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(data=data, x='drug_dose', y=ft, hue='drug_name', palette='colorblind',  showfliers=False)
        rc_params = {
            'font.sans-serif': "Arial",  # just an example
            'svg.fonttype': 'none',
            }
        with plt.rc_context(rc_params):
            plt.savefig(savefig/'{}.svg'.format(ft))
        plt.close()