#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:51:04 2020

@author: em812
"""
import pandas as pd
import pdb

def select_feature_set(feat, align_bluelight, path_to_set, append_to_names=None):

    cols = pd.read_csv(path_to_set).values.flatten()

    if align_bluelight:
        bluelight = ['prestim', 'bluelight', 'poststim']
        cols = ['_'.join([col, blue]) for col in cols for blue in bluelight]

    if append_to_names is not None:
        cols = ['_'.join([col, a]) for col in cols for a in append_to_names]

    feat = feat[[col for col in cols if col in feat]]

    return feat

#%% Read files Dec2019 screen
def read_only_feat_set(feat_file, feat_set, align_bluelight=False,
                       append_to_names=None, comment='#'):
    from moaclassification.helper import get_feat_set

    cols = get_feat_set(feat_set, align_bluelight=align_bluelight)

    if append_to_names is not None:
        cols = ['_'.join([col, a]) for col in cols for a in append_to_names]

    header = pd.read_csv(feat_file, nrows=1, comment='#').columns
    cols = [col for col in cols if col in header]

    feat = pd.read_csv(feat_file, usecols=cols, comment='#')

    return feat

def read_files(
        feat_file, fname_file, metadata_file,
        moa_file=None, comment='#'
        ):
    feat = pd.read_csv(feat_file, comment='#')
    fname = pd.read_csv(fname_file, comment='#')

    meta = pd.read_csv(metadata_file, index_col=None)
    meta.loc[meta['drug_type'].isna(), 'drug_type'] = 'NoCompound'


    if moa_file is not None:
        moa = pd.read_csv(moa_file, index_col=None)
    else:
        moa = None

    return feat, fname, meta, moa


#%% Build metadata and matching features dfs
def build_meta_and_matching_feat_dfs(feat, fname, meta, align_bluelight=True):
    from tierpsytools.read_data.hydra_metadata import \
        read_hydra_metadata, align_bluelight_conditions

    feat, meta = read_hydra_metadata(feat, fname, meta, add_bluelight=True)

    if align_bluelight:
        feat, meta = align_bluelight_conditions(feat, meta, how = 'outer')
    print('1: ', feat.shape)
    return feat, meta

def choose_bluelight_conditions(feat, meta, align_bluelight, choose_bluelight):
    if not isinstance(choose_bluelight, list):
        raise ValueError('choose_bluelight must be a list.')
    if align_bluelight:
        drop_bluelight = set(['prestim','bluelight','poststim']).difference(set(choose_bluelight))
        meta = meta[[col for col in meta.columns if not any([blue in col for blue in drop_bluelight]) ]]
        feat = feat[[col for col in feat.columns if not any([blue in col for blue in drop_bluelight]) ]]
    else:
        meta = meta[meta['bluelight'].isin(choose_bluelight)]
        feat = feat.loc[meta.index, :]

    return feat, meta

def average_by_dose(feat, meta, align_bluelight=None, groupby=None):
    if align_bluelight is not None:
        if align_bluelight:
            groupbykeys = ['worm_strain', 'drug_type', 'imaging_plate_drug_concentration']
        else:
            groupbykeys = ['worm_strain', 'drug_type', 'imaging_plate_drug_concentration', 'bluelight']
    elif groupby is not None:
        groupbykeys=groupby
    else:
        raise ValueError('Must define align_bluelight or groupby input')

    feat = feat.groupby(by=[meta[key] for key in groupbykeys]).mean()
    meta = meta.drop_duplicates(subset=groupbykeys)
    meta = meta.set_index(groupbykeys).loc[feat.index]

    meta = meta.reset_index(drop=False)
    feat = feat.reset_index(drop=True)

    return feat, meta


def encode_categorical_variable(feat, variable, base_name=None, encoder=None):
    from sklearn.preprocessing import OneHotEncoder

    if encoder is None:
        encoder = OneHotEncoder(sparse=False)
    if isinstance(variable, pd.Series):
        if base_name is None:
            base_name = variable.name
        variable = variable.values

    encoded_ft = encoder.fit_transform(variable.reshape(-1,1))
    if len(encoded_ft.shape)==1:
        if base_name is None:
            base_name = 'encoded_feature'
        feat.insert(0, base_name, encoded_ft)
    else:
        if base_name is None:
            names = encoder.categories_[0]
        else:
            names = ['_'.join([base_name, ctg]) for ctg in encoder.categories_[0]]
        for col, names in enumerate(names):
            feat.insert(0, names, encoded_ft[:, col])
    return feat


def add_moa_info(meta, moa):
    # add the MOA information to the metadata
    if 'CSN' in moa:
        drug_name_col = 'CSN'
    elif 'drug_name' in moa:
        drug_name_col = 'drug_name'
    moa = moa.rename(columns={drug_name_col: "drug_type"})

    meta = meta.join(
        moa[['MOA_general', 'MOA_specific', 'drug_type', 'MOA_group']].set_index('drug_type'),
        on='drug_type')

    # add drug_dose column where dose for DMSO and NoCompound is 0.0
    meta.insert(0, 'drug_dose', meta['imaging_plate_drug_concentration'].astype(float))
    meta.loc[meta['drug_type'].isin(['DMSO', 'NoCompound']), 'drug_dose'] = 0.0
    return meta

def drop_drug_names(meta, names=['NoCompound'], feat=None):
    # Remove NoCompound and DMSO
    if feat is not None:
        feat = feat[~meta['drug_type'].isin(names)]
        print('remove drug names {}: '.format(names), feat.shape)
    meta = meta[~meta['drug_type'].isin(names)]

    if feat is not None:
        return feat, meta
    else:
        return meta

def select_tierpsy_feature_set(feat, feat_set, align_bluelight, append_to_names=None):
    from moaclassification.helper import get_feat_set

    cols = get_feat_set(feat_set, align_bluelight=align_bluelight)

    if append_to_names is not None:
        cols = ['_'.join([col, a]) for col in cols for a in append_to_names]

    feat = feat[[col for col in cols if col in feat]]

    return feat

#%%
# Remove bad wells and preprocess features
def drop_feat_groups(feat, keywords):
    # Remove features by keywords
    drop_ft = [col for col in feat.columns if any([key in col for key in keywords])]
    feat = feat[feat.columns.difference(drop_ft)]
    print('drop feat groups: ', feat.shape)
    return feat

def remove_bad_wells(meta, bad_well_cols=None, feat=None):
    # remove bad wells of any type
    if bad_well_cols is None:
        bad_well_cols = [col for col in meta.columns if 'is_bad' in col]

    bad = meta[bad_well_cols].any(axis=1)

    if feat is not None:
        feat = feat.loc[~bad,:]
        print('remove bad wells: ', feat.shape)
    meta = meta.loc[~bad,:]

    if feat is not None:
        return feat, meta
    else:
        return meta

def remove_missing_bluelight_conditions(meta, feat=None):
    # remove wells missing blueligth conditions
    imgst_cols = [col for col in meta.columns if 'imgstore_name' in col]
    bad = meta[imgst_cols].isna().any(axis=1)

    if feat is not None:
        feat = feat.loc[~bad,:]
        print('remove missing bluelight: ', feat.shape)
    meta = meta.loc[~bad,:]

    if feat is not None:
        return feat, meta
    else:
        return meta

def remove_samples_by_n_nan(feat, meta, threshold=7000):
    # remove wells with too many nans
    if threshold<1:
        threshold = int(threshold*feat.shape[1])
    meta = meta[feat.isna().sum(axis=1)<threshold]
    feat = feat[feat.isna().sum(axis=1)<threshold]
    print('remove samples by n nan: ', feat.shape)
    return feat, meta

def remove_ft_by_n_nan(feat, threshold):
    if threshold<1:
        threshold = int(threshold*feat.shape[0])
    # remove features with over 10% nans
    feat = feat.loc[:, feat.isna().sum(axis=0)<threshold]
    print('remove ft by n nan: ', feat.shape)
    return feat

def impute_nans(feat, strain):
    # fill in remaining nans with mean values of cv features for each strain separately
    feat = [x for _,x in feat.groupby(by=strain, sort=True)]
    for i in range(len(feat)):
        feat[i] = feat[i].fillna(feat[i].mean())
    feat = pd.concat(feat, axis=0).sort_index()
    return feat

#%%
# Select cv and test set
def cv_test_split(feat, meta, train_file, test_file):
    train_set = pd.read_csv(train_file)
    test_set = pd.read_csv(test_file)

    # Select cv and test set
    meta_test = meta[meta['drug_type'].isin(test_set['drug_type'].values)]
    meta_train = meta[meta['drug_type'].isin(train_set['drug_type'])]
    print('n samples train={}, test={}'.format(meta_train.shape[0], meta_test.shape[0]))
    return feat.loc[meta_train.index, :], meta_train, \
        feat.loc[meta_test.index, :], meta_test

#%% One function
def preprocess_main(feat_file, fname_file, metadata_file,
        moa_file=None, test_file=None, train_file=None,
        align_bluelight=True,
        average_dose=False,
        add_bluelight_id=False,
        choose_strains=None,
        add_strain_id=False,
        choose_bluelight=None,
        compounds_to_drop=None,
        select_tierpsy_set=None,
        feat_groups_to_drop=None,
        remove_bad=True, bad_well_cols=None,
        remove_missing_bluelight=True,
        sample_max_nan_threshold=0.8,
        feat_max_nan_threshold=0.1,
        split_cv_test=True,
        impute_nan_values=True):

    # Read files Dec2019 screen
    feat, fname, meta, moa = read_files(
        feat_file, fname_file, metadata_file,
        moa_file=moa_file,
        comment='#')

    # Build metadata and matching features dfs
    feat, meta = build_meta_and_matching_feat_dfs(
        feat, fname, meta, align_bluelight=align_bluelight)

    if choose_strains is not None:
        feat = feat[meta['worm_strain'].isin(choose_strains)]
        meta = meta[meta['worm_strain'].isin(choose_strains)]

    if choose_bluelight is not None:
        feat, meta = choose_bluelight_conditions(feat, meta, align_bluelight, choose_bluelight)

    if average_dose:
        feat, meta = average_by_dose(feat, meta, align_bluelight=align_bluelight)

    if select_tierpsy_set is not None:
        feat = select_tierpsy_feature_set(feat, select_tierpsy_set, align_bluelight)

    if add_bluelight_id:
        if align_bluelight:
            raise ValueError('Cannnot add bluelight_id in aligned features dataframe.')
        feat = encode_categorical_variable(feat, meta['bluelight'])

    if add_strain_id:
        feat = encode_categorical_variable(feat, meta['worm_strain'].values,
                                           base_name='strain')
    if moa is not None:
        meta = add_moa_info(meta, moa)

    if compounds_to_drop is not None:
        feat, meta = drop_drug_names(meta, names=compounds_to_drop, feat=feat)

    # Remove bad wells and preprocess features
    if feat_groups_to_drop is not None:
        feat = drop_feat_groups(feat, feat_groups_to_drop)
    if remove_bad:
        feat, meta = remove_bad_wells(
            meta, bad_well_cols=bad_well_cols, feat=feat)
    if remove_missing_bluelight:
        if align_bluelight:
            feat, meta = remove_missing_bluelight_conditions(meta, feat=feat)
    if sample_max_nan_threshold is not None:
        feat, meta = remove_samples_by_n_nan(
            feat, meta, threshold=sample_max_nan_threshold)
    if feat_max_nan_threshold is not None:
        feat = remove_ft_by_n_nan(
            feat, feat_max_nan_threshold)

    if split_cv_test:
        if train_file is None or test_file is None:
            raise ValueError(
                'Test and train set files need to be provided to make split.')
        feat, meta, feat_test, meta_test = \
            cv_test_split(feat, meta, train_file, test_file)

    if impute_nan_values:
        if align_bluelight:
            feat = impute_nans(feat, meta['worm_strain'])
            if split_cv_test:
                feat_test = impute_nans(feat_test, meta_test['worm_strain'])
        else:
            feat = [impute_nans(feat.loc[x.index, :], x['worm_strain'])
                    for blue,x in meta.groupby(by='bluelight')]
            feat = pd.concat(feat).sort_index()
            if split_cv_test:
                feat_test = [impute_nans(feat_test.loc[x.index, :], x['worm_strain'])
                        for blue,x in meta_test.groupby(by='bluelight')]
                feat_test = pd.concat(feat_test).sort_index()

    if split_cv_test:
        return feat, meta, feat_test, meta_test
    else:
        return feat, meta

def preprocess_main_bluewindows(
        feat_file, fname_file, metadata_file,
        win_feat_file, win_fname_file,
        moa_file=None, test_file=None, train_file=None,
        align_bluelight=True,
        average_dose=False,
        compounds_to_drop=None,
        feat_groups_to_drop=None,
        remove_bad=True, bad_well_cols=None,
        remove_missing_bluelight=True,
        sample_max_nan_threshold=0.8,
        feat_max_nan_threshold=0.1,
        split_cv_test=True,
        impute_nan_values=True):

    feat, meta = preprocess_main(
        feat_file, fname_file, metadata_file, moa_file=moa_file,
        align_bluelight=align_bluelight,
        compounds_to_drop=compounds_to_drop,
        feat_groups_to_drop=feat_groups_to_drop,
        choose_bluelight=['prestim','poststim'],
        remove_missing_bluelight=True,
        sample_max_nan_threshold=None, feat_max_nan_threshold=None,
        split_cv_test=False, impute_nan_values=False)

    win_feat, win_meta = preprocess_main(
        win_feat_file, win_fname_file, metadata_file, moa_file=moa_file,
        align_bluelight=align_bluelight,
        compounds_to_drop=compounds_to_drop,
        feat_groups_to_drop=feat_groups_to_drop,
        choose_bluelight=['bluelight'],
        remove_missing_bluelight=True,
        sample_max_nan_threshold=None, feat_max_nan_threshold=None,
        split_cv_test=False, impute_nan_values=False)

    if align_bluelight:
        if remove_missing_bluelight:
            how = 'inner'
        else:
            how = 'outer'
        meta = pd.merge(meta.reset_index(), win_meta.reset_index(),
                        on=['date_yyyymmdd','imaging_plate_id','well_name'],
                        suffixes=('','_win'), how=how)
        del win_meta
        feat = feat.loc[meta['index'].values, :].reset_index(drop=True)
        win_feat = win_feat.loc[meta['index_win'].values, :].reset_index(drop=True)
        feat = pd.concat([feat, win_feat], axis=1)
        del win_feat

        meta = meta[[col for col in meta.columns if not col.endswith('_win')]]
    else:
        meta = pd.concat([meta, win_meta], axis=0)
        feat = pd.concat([feat, win_feat], axis=0)
        meta = meta.reset_index(drop=True)
        feat = feat.reset_index(drop=True)
        del win_meta
        del win_feat

    if average_dose:
        feat, meta = average_by_dose(feat, meta, align_bluelight=align_bluelight)


    if sample_max_nan_threshold is not None:
        feat, meta = remove_samples_by_n_nan(feat, meta, threshold=0.8)
    if feat_max_nan_threshold is not None:
        feat = remove_ft_by_n_nan(feat, 0.1)

    if split_cv_test:
        if train_file is None or test_file is None:
            raise ValueError(
                'Test and train set files need to be provided to make split.')
        feat, meta, feat_test, meta_test = cv_test_split(feat, meta, train_file, test_file)

    if impute_nan_values:
        if align_bluelight:
            feat = impute_nans(feat, meta['worm_strain'])
            if split_cv_test:
                feat_test = impute_nans(feat_test, meta_test['worm_strain'])
        else:
            feat = [impute_nans(feat.loc[x.index, :], x['worm_strain'])
                    for blue,x in meta.groupby(by='bluelight')]
            feat = pd.concat(feat).sort_index()
            if split_cv_test:
                feat_test = [impute_nans(feat_test.loc[x.index, :], x['worm_strain'])
                        for blue,x in meta_test.groupby(by='bluelight')]
                feat_test = pd.concat(feat_test).sort_index()


    if split_cv_test:
        return feat, meta, feat_test, meta_test
    else:
        return feat, meta
