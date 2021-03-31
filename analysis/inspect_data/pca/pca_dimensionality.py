#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:41:33 2020

@author: em812
"""

from pathlib import Path
import pandas as pd
from moaclassification import INPUT_DIR
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from numpy import cumsum

#%% Input for aligned bluelight
align_blue = True

root_in = Path(INPUT_DIR)

# Input directory
data_file = root_in/ 'features.csv'
meta_file = root_in/ 'metadata.csv'

saveroot = Path().cwd() / 'figures'
saveroot.mkdir(exist_ok=True)

#%% Read and preprocess data
feat = pd.read_csv(data_file, index_col=None)
meta = pd.read_csv(meta_file, index_col=None)
meta  = meta[['MOA_group', 'drug_type', 'drug_dose', 'worm_strain',
              'MOA_general', 'MOA_specific', 'date_yyyymmdd']]

# Impute nans
feat = feat.fillna(feat.mean())

# scale
feat = scale(feat)

#%% PCA
pca = PCA(n_components=None)
pca.fit(feat)

#%% plot variance explained
y = cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(5,3))
plt.plot(cumsum(pca.explained_variance_ratio_), label='_nolegend')
for n,c in zip([512, 1024, 2048],['red','green', 'black']):
    plt.scatter(n, y[n], marker='x', c=c, label='{} components - {:.2f}%'.format(n, y[n]*100))
plt.legend()
plt.xlabel('number of components')
plt.ylabel('total variance explained')
plt.tight_layout()
plt.savefig('variance_explained.png')