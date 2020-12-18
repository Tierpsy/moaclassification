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
from tierpsytools.feature_processing.filter_features import select_feat_set
from tierpsytools.analysis.clustering_tools import hierarchical_purity
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.cluster.hierarchy import linkage
plt.rcParams['svg.fonttype'] = 'none'

#%% Input for aligned bluelight
method='complete'
metric='cosine'

align_blue = True

root = Path('/Users/em812/Documents/Workspace/Drugs/StrainScreens/SyngentaN2')

# Input directory
root_in = root / 'create_datasets_both_screens' / 'all_data'
root_in = root_in / 'filtered_align_blue={}_average_dose=False_feat=all'.format(align_blue)

data_file = root_in/ 'features.csv'
meta_file = root_in/ 'metadata.csv'

# split directory
root_split = root/ 'test_set'/'univariate_LMM'/'filtered_data' / 'curated'

train_file = root_split / 'train_compounds.csv'
test_file = root_split / 'test_compounds.csv'
novel_file = root_split / 'novelty_compounds.csv'

# MOa file
moa_file = '/Users/em812/Data/Drugs/StrainScreens/AllCompoundsMoA.csv'

# LUT file
lut_file = '/Users/em812/OneDrive - Imperial College London/share/SyngentaPaper/MOA_colors.csv'

saveroot = Path().cwd() / 'figures'
saveroot.mkdir(exist_ok=True)

#%% Read and preprocess data
train_compounds = pd.read_csv(train_file)
test_compounds = pd.read_csv(test_file)
novel_compounds = pd.read_csv(novel_file)

feat = pd.read_csv(data_file, index_col=None)
meta = pd.read_csv(meta_file, index_col=None)
meta  = meta[['MOA_group', 'drug_type', 'drug_dose', 'worm_strain',
              'MOA_general', 'MOA_specific', 'date_yyyymmdd']]
moa = pd.read_csv(moa_file)

# Combine classes 3-4
meta.loc[meta['MOA_group']==4, 'MOA_group'] = 3
moa.loc[moa['MOA_group']==4, 'MOA_group'] = 3

moamapper = dict(moa[['MOA_group', 'MOA_general']].drop_duplicates(subset=['MOA_general']).values)
meta['MOA_general'] = meta['MOA_group'].map(moamapper)

# Keep only train test compounds
meta = meta[meta['drug_type'].isin(train_compounds['drug_type'].to_list()+
                                   test_compounds['drug_type'].to_list() #+
                                   #novel_compounds['drug_type'].to_list()
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
# Get average dose for DMSO points
# dmso_ids = meta['drug_type']=='DMSO'
# feat_dmso = feat[dmso_ids]
# meta_dmso = meta[dmso_ids]

# feat_dmso = [feat_dmso.sample(frac=0.3).mean() for i in range(20)]
# feat_dmso = pd.concat(feat_dmso, axis=1).T
# meta_dmso = pd.concat([meta.iloc[[0],:] for i in range(20)], axis=0)
# meta_dmso.index = feat_dmso.index

dmso_ids = meta['drug_type']=='DMSO'
feat_dmso = feat[dmso_ids]
meta_dmso = meta[dmso_ids]

feat_dmso = feat_dmso.groupby(
    by=meta_dmso['date_yyyymmdd']).apply(lambda x: x.sample(n=11).mean())

meta_dmso = [meta_dmso.loc[meta_dmso['date_yyyymmdd']==date, :].iloc[0,:]
              for date in feat_dmso.index]
meta_dmso = pd.concat(meta_dmso, axis=1).T

meta_dmso = meta_dmso.reset_index(drop=True)
feat_dmso = feat_dmso.reset_index(drop=True)

# Get average doses for remaining compounds
feat_cols = feat.columns

feat = pd.concat([feat[~dmso_ids], meta.loc[~dmso_ids, ['drug_type', 'drug_dose']]], axis=1)
feat = feat.groupby(by=['drug_type', 'drug_dose']).mean()

meta = meta.loc[~dmso_ids].groupby(by=['drug_type', 'drug_dose']).agg(lambda x: x.iloc[0])
feat.index = ['_'.join([str(x) for x in a]) for a in feat.index]
meta.index = feat.index

# Combine DMSO with other compounds
feat = pd.concat([feat, feat_dmso])
meta = pd.concat([meta, meta_dmso])

# Scale features
feat = pd.DataFrame(scale(feat), index=feat.index, columns=feat.columns)

#%% Get row linkage

row_linkage = linkage(feat, method=method, metric=metric)

#%% Plot heatmap
savefig = saveroot/'figure2_clustermap'
savefig.mkdir(exist_ok=True)

labels = meta['MOA_general']
labels.name = 'MOA'

lut = pd.read_csv(lut_file, header=None)
lut = dict(lut.values)
row_colors = pd.DataFrame(labels)['MOA'].map(lut)

legend_TN = [mpatches.Patch(color=c, label=l) for l,c in lut.items()]

g = sns.clustermap(feat, vmin=-1, vmax=1, method=method,
                    row_linkage=row_linkage, row_colors=row_colors,
                    figsize=(15,10), xticklabels=False, yticklabels=False,
                    cbar_kws={'label': 'znorm'})
g.ax_heatmap.xaxis.axes.xaxis.set_label_text('features')
g.ax_heatmap.yaxis.axes.yaxis.set_label_text('all doses of all compounds')

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
