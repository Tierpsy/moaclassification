#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Perform univariate statistical tests for the chosen compounds and correct for
multiple comparisons, taking into account all the features at all the doses.

Created on Fri Mar 12 21:08:47 2021

@author: em812
"""
from pathlib import Path
import pandas as pd
import numpy as np
from tierpsytools.drug_screenings.filter_compounds import compounds_with_low_effect_univariate
from moaclassification import INPUT_DIR
from functools import reduce
import pdb
import statsmodels.formula.api as smf
import json


DRUGS_TO_PLOT = {
    #3: ['SY0376', 'SY0354', 'SY0353'],
    # 6: ['SY0622', 'SY0641', 'SY0612'],
    10: ['SY1048', 'SY1081', 'SY1021'],
    # 11: ['SY1154', 'SY1140', 'SY1124'],
    17: ['SY1793', 'SY1786', 'SY1713']
    }

FTS_TO_TEST = ['curvature_hips_abs_50th', 'angular_velocity_head_tip_abs_50th']

#%% Input
# Paths to data
data_file = Path(INPUT_DIR) / 'long_format'/ 'features_long_format.csv'
meta_file = Path(INPUT_DIR) / 'long_format'/ 'metadata_long_format.csv'
moa_file = Path(INPUT_DIR) / 'AllCompoundsMoA.csv'

saveroot = Path().cwd() / 'dose_response_curves'
saveroot.mkdir(exist_ok=True)

#%% Read and preprocess date
feat = pd.read_csv(data_file, index_col=None)
meta = pd.read_csv(meta_file, index_col=None)
moa = pd.read_csv(moa_file)

# Keep only features of interest
feat = feat[FTS_TO_TEST]

# Keep only compounds of interest
mask = meta['drug_type'].isin(reduce(lambda x,y: x+y, list(DRUGS_TO_PLOT.values())+[['DMSO']]))
feat = feat[mask]
meta = meta[mask]

# Add the market names to the metadata
namemapper = dict(moa[['CSN', 'drug_name']].values)
meta['drug_name'] = meta['drug_type'].map(namemapper)

#%% Statistical tests
all_res = {}
for drug in reduce(lambda x,y: x+y, list(DRUGS_TO_PLOT.values())):
    mask = meta['drug_type'].isin(['DMSO', drug])
    X = feat[mask]
    drug_dose = meta[mask]['drug_dose']
    random_effect = meta[mask]['date_yyyymmdd'].astype(str)

    X = X.assign(drug_dose=drug_dose).assign(random_effect=random_effect)

    # select only the control points that belong to groups that have
    # non-control members
    groups = np.unique(random_effect[drug_dose!=0])
    X = X[np.isin(random_effect, groups)]

    # Intitialize pvals as series or dataframe
    # (based on the number of comparisons per feature)
    res = pd.DataFrame(columns=['pval', 'coef'], index=FTS_TO_TEST, dtype=float)

    ## Fit LMMs for each feature
    for ft in FTS_TO_TEST:
        data = X[[ft, 'drug_dose', 'random_effect']].dropna(axis=0)

        data = pd.concat(
            [x for _,x in data.groupby(by=['drug_dose', 'random_effect']) if x.shape[0]>1]
            )

        # Define LMM
        md = smf.mixedlm("{} ~ drug_dose".format(ft), data,
                         groups=data['random_effect'].astype(str),
                         re_formula="")
        # Fit LMM
        try:
            mdf = md.fit()
        except:
            res.loc[ft, 'pval'] = np.nan
            res.loc[ft, 'coef'] = np.nan
        else:
            res.loc[ft, 'pval'] = mdf.pvalues['drug_dose']
            res.loc[ft, 'coef'] = mdf.fe_params['drug_dose']

    all_res[drug] = res

all_res = pd.concat(
    [all_res[drug].assign(drug=drug).set_index('drug', append=True).reorder_levels(['drug', 0])
     for drug in all_res.keys()], axis=0)

all_res.to_csv(saveroot/'stats.csv')
