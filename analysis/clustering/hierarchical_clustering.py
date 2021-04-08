#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 14:41:33 2020

@author: em812
"""

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import scale
import numpy as np
from tierpsytools.preprocessing.filter_data import select_feat_set
from tierpsytools.analysis.clustering_tools import hierarchical_purity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage
from sklearn.model_selection import StratifiedKFold
from moaclassification import INPUT_DIR

DRUGS_TO_HIGHLIGHT = ['SY1048', 'SY1081', 'SY1021', 'SY1793', 'SY1786', 'SY1713']
# lut_h = dict(zip(['SY1048', 'SY1081', 'SY1021', 'other', 'SY1793', 'SY1786', 'SY1713'],
#                  ['#8c510a', '#d8b365', '#f6e8c3', '#f5f5f5', '#c7eae5', '#5ab4ac', '#01665e']))

plt.rcParams['svg.fonttype'] = 'none'

#%% Input for aligned bluelight
method='complete'
metric='cosine'

align_blue = True

# Input directory
data_file = Path(INPUT_DIR) / 'features.csv'
meta_file = Path(INPUT_DIR) / 'metadata.csv'

# split directory
train_file = Path(INPUT_DIR) / 'split' / 'train_compounds.csv'
test_file = Path(INPUT_DIR) / 'split' / 'test_compounds.csv'

# MOa file
moa_file = Path(INPUT_DIR) / 'AllCompoundsMoA.csv'

# LUT file
lut_file = Path(INPUT_DIR) / 'MOA_colors.csv'
lut_h_file = Path(INPUT_DIR) / 'highlighted_drugs_colors.csv'

saveroot = Path().cwd() / 'figures'
saveroot.mkdir(exist_ok=True)

#%% Read and preprocess data
train_compounds = pd.read_csv(train_file)
test_compounds = pd.read_csv(test_file)

feat = pd.read_csv(data_file, index_col=None)
meta = pd.read_csv(meta_file, index_col=None)
meta  = meta[['MOA_group', 'drug_type', 'drug_dose', 'worm_strain',
              'MOA_general', 'MOA_specific', 'date_yyyymmdd']]
moa = pd.read_csv(moa_file)

moamapper = dict(moa[['MOA_group', 'MOA_general']].drop_duplicates(subset=['MOA_general']).values)
meta['MOA_general'] = meta['MOA_group'].map(moamapper)

# Keep only train test compounds
meta = meta[meta['drug_type'].isin(train_compounds['drug_type'].to_list()+
                                   test_compounds['drug_type'].to_list()
                                   )]
feat = feat.loc[meta.index]

# Impute nans
if align_blue:
    feat = feat.fillna(feat.mean())
else:
    means_cv = {blue: x.mean() for blue,x in feat.groupby(by=meta['bluelight'])}
    feat = [x.fillna(means_cv[blue])
            for blue,x in feat.groupby(by=meta['bluelight'])]
    feat = pd.concat(feat).sort_index()

# Choose tierpsy256
feat = select_feat_set(feat, tierpsy_set_name='tierpsy_256', append_bluelight=align_blue)

#%% Get average doses
# Get the DMSO points
dmso_ids = meta['drug_type']=='DMSO'
meta_dmso = meta[dmso_ids]

splitter = StratifiedKFold(n_splits=6)
for i,(_, idx) in enumerate(splitter.split(meta_dmso, meta_dmso['date_yyyymmdd'])):
    dfidx = meta_dmso.index[idx]
    meta_dmso.loc[dfidx, 'drug_type'] = 'DMSO_{}'.format(i)

meta[dmso_ids] = meta_dmso

# Get average doses for remaining compounds
feat_cols = feat.columns

feat = pd.concat([feat, meta[['drug_type', 'drug_dose']]], axis=1)
feat = feat.groupby(by=['drug_type', 'drug_dose']).mean()

meta = meta.groupby(by=['drug_type', 'drug_dose']).agg(lambda x: x.iloc[0])
meta = meta.reset_index(drop=False)
feat.index = ['_'.join([str(x) for x in a]) for a in feat.index]
meta.index = feat.index

# Scale features
feat = pd.DataFrame(scale(feat), index=feat.index, columns=feat.columns)

#%% Get row linkage

row_linkage = linkage(feat, method=method, metric=metric)

#%% Plot heatmap
savefig = saveroot/'figure2_clustermap'
savefig.mkdir(exist_ok=True)

# Get row colors that show MOA and mathcing legend data
labels = meta['MOA_general']
labels.name = 'MOA'

lut = pd.read_csv(lut_file, header=None)
lut = dict(lut.values)
row_colors = pd.DataFrame(labels)['MOA'].map(lut)

legend_TN = [mpatches.Patch(color=c, label=l) for l,c in lut.items()]

# Get row colors that show highlighted drugs
highlight_labels = meta['drug_type']
highlight_labels[~highlight_labels.isin(DRUGS_TO_HIGHLIGHT)] = 'other'
lut_h = pd.read_csv(lut_h_file)
lut_h = dict(lut_h.values)

highlights = pd.DataFrame(highlight_labels)['drug_type'].map(lut_h)

# Plot the heatmap
g = sns.clustermap(feat, vmin=-1, vmax=1, method=method,
                    row_linkage=row_linkage, row_colors=[highlights, row_colors],
                    figsize=(15,10), xticklabels=False, yticklabels=False,
                    cbar_kws={'label': 'znorm'})
g.ax_heatmap.xaxis.axes.xaxis.set_label_text('features')
g.ax_heatmap.yaxis.axes.yaxis.set_label_text('all doses of all compounds')

# Make legend for MOA colors
l2=g.ax_heatmap.legend(loc='center left', bbox_to_anchor=(1.1,0.5),
                        handles=legend_TN, frameon=True, fontsize=14)
l2.set_title(title='MOA groups',prop={'size':14})
plt.tight_layout()
rc_params = {
    'font.sans-serif': "Arial",  # just an example
    'svg.fonttype': 'none',
    }
with plt.rc_context(rc_params):
    plt.savefig(savefig/'heatmap_average_doses_align_blue=True_{}.png'.format(method))
    plt.savefig(savefig/'heatmap_average_doses_align_blue=True_{}.svg'.format(method))
    plt.savefig(savefig/'heatmap_average_doses_align_blue=True_{}.pdf'.format(method))
plt.close()

  #%% Get the cluster purity
savefig = saveroot/'figure2_cluster_purity'
savefig.mkdir(exist_ok=True)

dist, n_clust, purity, rand_purity = hierarchical_purity(
    feat, meta['MOA_group'].values, linkage_matrix=row_linkage, n_random=1000)

plt.figure()
plt.plot(dist, purity, label='Phenotypic clustering', c='red', linewidth=2.0)
plt.errorbar(dist, np.mean(rand_purity, axis=1), yerr=np.std(rand_purity, axis=1), label='Random clusters', c='grey')
plt.ylabel('Cluster purity')
plt.xlabel('Hierachical clustering distance')
plt.legend()
with plt.rc_context(rc_params):
    plt.savefig(savefig/'distance-purity.svg')
