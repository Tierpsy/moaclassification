#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:57:47 2021

@author: em812
"""


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest
from moaclassification import INPUT_DIR
import pdb
from tierpsytools.preprocessing.scaling_class import scalingClass
from tierpsytools.preprocessing.filter_data import filter_nan_inf
from textwrap import fill
from mpl_toolkits.mplot3d import Axes3D, proj3d
from sklearn.decomposition import PCA

moas_to_plot = [3,7]
colors = ['#ff7f00', '#6a3d9a']

rc_params = {
            'font.sans-serif': "Arial",  # just an example
            'svg.fonttype': 'none',
            }
#%% Input
# Paths to data
data_file = Path(INPUT_DIR) / 'features.csv'
meta_file = Path(INPUT_DIR) / 'metadata.csv'

s_data_file = Path(INPUT_DIR) / 'get_smoothed_balanced_data'/ 'smoothed_balanced_data' / 'features_cv.csv'
s_meta_file = Path(INPUT_DIR) / 'get_smoothed_balanced_data'/ 'smoothed_balanced_data' / 'metadata_cv.csv'

moa_file = Path(INPUT_DIR) / 'AllCompoundsMoA.csv'

saveto = Path().cwd() / 'figures'
saveto.mkdir(exist_ok=True)

#%% Read and preprocess date
sfeat = pd.read_csv(s_data_file, index_col=None)
smeta = pd.read_csv(s_meta_file, index_col=None)

feat = pd.read_csv(data_file, index_col=None)
meta = pd.read_csv(meta_file, index_col=None)
moa = pd.read_csv(moa_file)

# keep only cv set drugs
mask = meta['drug_type'].isin(smeta['drug_type'].unique())
meta = meta[mask]
feat = feat[mask]

# Add the market names to the metadata
namemapper = dict(moa[['CSN', 'drug_name']].values)
meta['drug_name'] = meta['drug_type'].map(namemapper)
smeta['drug_name'] = smeta['drug_type'].map(namemapper)

# Choose only prestimulus
cols = [c for c in feat.columns if c.endswith('_prestim')]
feat = feat[cols]
sfeat = sfeat[cols]

# Filter nans
feat = filter_nan_inf(feat, 0.5, axis=1)
feat = filter_nan_inf(feat, 0.05, axis=0)
feat = feat.fillna(feat.mean())

# Remove DMSO
mask = meta['drug_type']!='DMSO'
meta = meta[mask]
feat = feat[mask]
mask = smeta['drug_type']!='DMSO'
smeta = smeta[mask]
sfeat = sfeat[mask]

#%% Keep only the moas of interest
# mask = meta['drug_type'].isin(DRUGS)
# meta = meta[mask]
# feat = feat[mask]

# make mapper to get drug names
meta['name_'] = meta[['MOA_group', 'drug_name']].apply(lambda x: ' - '.join(map(str, x.values)), axis=1)
namemapper = dict(meta[['drug_type', 'name_']].values)

moamapper = dict(meta[['MOA_group', 'MOA_general']].drop_duplicates(subset=['MOA_group']).values)

#%% Scale features
scaler = scalingClass()

feat = scaler.fit_transform(feat)
sfeat = scaler.fit_transform(sfeat)

#%% PCA colored by moa
pca = PCA(n_components=3)
Y = pca.fit_transform(feat)
sY = pca.transform(sfeat)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('original data')
#    ax.set_aspect('equal')
for moa,c in zip(moas_to_plot, colors):
    mask = meta['MOA_group'].isin([moa]).values
    ax.scatter(*Y[mask].T, s=7, c=c, label=moamapper[moa])
plt.legend()
ax.set_xlabel('PC 1', fontsize=10)
ax.set_ylabel('PC 2', fontsize=10)
ax.set_zlabel('PC 3', fontsize=10)
f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
ax.legend(loc="lower left", bbox_to_anchor=f(100,70,70),
          bbox_transform=ax.transData)
plt.tight_layout()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
zlim = ax.get_zlim()
with plt.rc_context(rc_params):
    plt.savefig(saveto/'PCA.svg')

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('bootstrap averages')
#    ax.set_aspect('equal')
for moa,c in zip(moas_to_plot, colors):
    mask = smeta['MOA_group'].isin([moa]).values
    ax.scatter(*sY[mask].T, s=7, c=c, label=moamapper[moa])
xlim = ax.set_xlim(xlim)
ylim = ax.set_ylim(ylim)
zlim = ax.set_zlim(zlim)
plt.legend()
ax.set_xlabel('PC 1', fontsize=10)
ax.set_ylabel('PC 2', fontsize=10)
ax.set_zlabel('PC 3', fontsize=10)
f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
ax.legend(loc="lower left", bbox_to_anchor=f(100,70,70),
          bbox_transform=ax.transData)
plt.tight_layout()
with plt.rc_context(rc_params):
    plt.savefig(saveto/'smoothed_PCA.svg')


#%% PCA colored by drug
#    ax.set_aspect('equal')
for moa in moas_to_plot:
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(moamapper[moa])
    mask = meta['MOA_group'].isin([x for x in moas_to_plot if x!=moa]).values
    ax.scatter(*Y[mask].T, s=7, c='grey', label='_nolegend_')
    for drug in meta.loc[meta['MOA_group'].isin([moa]), 'drug_type'].unique():
        mask = meta['drug_type'].isin([drug]).values
        ax.scatter(*Y[mask].T, s=7, label=drug)
    plt.legend()
    ax.set_xlabel('PC 1', fontsize=10)
    ax.set_ylabel('PC 2', fontsize=10)
    ax.set_zlabel('PC 3', fontsize=10)
    f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
    ax.legend(loc="lower left", bbox_to_anchor=f(100,100,0),
              bbox_transform=ax.transData)
    plt.tight_layout()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()
    with plt.rc_context(rc_params):
        plt.savefig(saveto/'moa={}_PCA.png'.format(moa))

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(' - '.join([moamapper[moa], 'bootstrap averages']))
    mask = smeta['MOA_group'].isin([x for x in moas_to_plot if x!=moa]).values
    ax.scatter(*sY[mask].T, s=7, c='grey', label='_nolegend_')
    for drug in smeta.loc[smeta['MOA_group'].isin([moa]), 'drug_type'].unique():
        mask = smeta['drug_type'].isin([drug]).values
        ax.scatter(*sY[mask].T, s=7, label=drug)
    plt.legend()
    ax.set_xlabel('PC 1', fontsize=10)
    ax.set_ylabel('PC 2', fontsize=10)
    ax.set_zlabel('PC 3', fontsize=10)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    f = lambda x,y,z: proj3d.proj_transform(x,y,z, ax.get_proj())[:2]
    ax.legend(loc="lower left", bbox_to_anchor=f(100,100,0),
              bbox_transform=ax.transData)
    plt.tight_layout()
    with plt.rc_context(rc_params):
        plt.savefig(saveto/'moa={}_sPCA.png'.format(moa))
