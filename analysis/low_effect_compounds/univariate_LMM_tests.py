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
from pathlib import Path
from tierpsytools.drug_screenings.filter_compounds import \
    compounds_with_low_effect_univariate
from moaclassification import INPUT_DIR

#%% Input
control = 'DMSO'

# Input directory
data_file = Path(INPUT_DIR) / 'features.csv'
meta_file = Path(INPUT_DIR) / 'metadata.csv'

saveto = Path().cwd() / 'results'
saveto.mkdir(parents=True, exist_ok=True)

#%% Read data
feat = pd.read_csv(data_file)
meta = pd.read_csv(meta_file)

# I don't impute nans, since the univariate tests can ingore nan values

#%% Univariate tests for every compound
signif_effect_drugs, low_effect_drugs, error_drugs, significant_df, pvals_df = \
    compounds_with_low_effect_univariate(
        feat, meta['drug_type'], drug_dose=meta['drug_dose'],
        random_effect=meta['date_yyyymmdd'],
        control=control, test='LMM', comparison_type='multiclass',
        multitest_method='fdr_by', fdr=0.01,
        ignore_names=None, return_pvals=True, n_jobs=-1)

pvals_df.to_csv(saveto/'pvals_LMM.csv')
significant_df.to_csv(saveto/'significant_LMM.csv')

pd.Series(low_effect_drugs, name='low_effect_drugs').to_csv(
    saveto/'low_effect_drugs.csv', index=None)
pd.Series(signif_effect_drugs, name='signif_effect_drugs').to_csv(
    saveto/'signif_effect_drugs.csv', index=None)
pd.Series(error_drugs, name='error_drugs').to_csv(
    saveto/'error_drugs.csv', index=None)

