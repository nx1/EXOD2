from astropy.units.quantity import SUBCLASS_SAFE_FUNCTIONS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from exod.utils.path import data_util, data_catalogue, data_simbad_stats
from astropy.table import Table
from exod.utils.simbad_classes import simbad_classifier

#Supress warnings
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

NULL20SPACES = '                    ' # 20 spaces because of byte encoding

def get_otype_col(df):
    return [col for col in df.columns if 'main_type' in col][0]

def tab2df(tab):
    """Return value counts of "otype" column."""
    df = tab.to_pandas()
    otype_col = get_otype_col(df)
    df[otype_col] = df[otype_col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x).str.strip()
    df = df.rename(columns={otype_col : 'otype'})
    return df

def value_counts_otypes(df):
    # Count the otypes.
    df = pd.DataFrame(df['otype'].value_counts())
    df.columns = ['object_count']
    return df

def calc_percentage(df, sumtotal):
    df['percentage'] = df['object_count'] / sumtotal * 100
    df['percentage'] = df['percentage'].round(2)
    df = df.reset_index()
    return df

def rename_cols(df):
    df.columns = ['otype', 'object_count', 'percentage']
    return df

def process_table(tab, sumtotal):
    df = tab2df(tab)
    df = value_counts_otypes(df)
    df = calc_percentage(df, sumtotal=sumtotal)
    df = rename_cols(df)
    df.sumtotal = sumtotal
    return df


def get_all_savepaths(dfs):
    savedir = data_simbad_stats
    all_savepaths = [savedir / f'{k}_otype_stats.csv' for k in dfs.keys()]
    all_savepaths.append(savedir / 'n_subtypes_vs_data_subset.csv')
    all_savepaths.append(savedir / 'n_subtypes_vs_data_subset_pivot.csv')
    all_savepaths.append(savedir / 'otype_stats_perc.csv')
    all_savepaths.append(savedir / 'otype_stats_count.csv')
    return all_savepaths
 
def load_dfs():
    print('Reading exod cat, dr14 and simbad tables...')
    #df_otypes   = pd.read_csv(data_util     / 'simbad_otypes_counts_27_08_24.csv')
    tab_exod_cat = Table.read(data_catalogue / 'EXOD_DR1_cat_unique.fits')
    tab_dr14     = Table.read(data_util      / '4xmmdr14slim_cmatch_simbad.fits')
    df_otypes    = pd.read_csv(data_util     / 'simbad_otype_counts_long.csv')
    df_otypes['percentage'] = df_otypes['object_count'] / df_otypes['object_count'].sum() * 100
    df_otypes['percentage'] = df_otypes['percentage'].round(2)
    df_otypes.sumtotal      = df_otypes['object_count'].sum()
    
    tab_exod_cat_simbad_matches_only = tab_exod_cat[~tab_exod_cat['simbad_main_type'].mask]
    tab_exod_cat_dr14_transient      = tab_exod_cat_simbad_matches_only[tab_exod_cat_simbad_matches_only['DR14_SC_VAR_FLAG'] == 1]
    tab_exod_cat_dr14_not_transient  = tab_exod_cat_simbad_matches_only[tab_exod_cat_simbad_matches_only['DR14_SC_VAR_FLAG'] == 0]
    tab_dr14                         = tab_dr14[tab_dr14['main_type'] != NULL20SPACES]
    tab_dr14_transient               = tab_dr14[(tab_dr14['SC_VAR_FLAG_1'] == 1) | (tab_dr14['SC_VAR_FLAG_2'] == 1)]
    
    # Get otype counts for each table
    n_dr14 = 692109
    df_dr14                        = process_table(tab_dr14,                        sumtotal=n_dr14)
    df_dr14_transient              = process_table(tab_dr14_transient,              sumtotal=n_dr14)
    
    df_exod_cat                    = process_table(tab_exod_cat,                    sumtotal=len(tab_exod_cat))
    df_exod_cat_dr14_transient     = process_table(tab_exod_cat_dr14_transient,     sumtotal=len(tab_exod_cat))
    df_exod_cat_dr14_not_transient = process_table(tab_exod_cat_dr14_not_transient, sumtotal=len(tab_exod_cat))
    
    
    dfs = {'SIMBAD'                     : df_otypes,
           'DR14 & SIMBAD'              : df_dr14,
           'DR14 & SIMBAD & TRANSIENT'  : df_dr14_transient,
           'EXOD FULL'                  : df_exod_cat,
           'EXOD & TRANSIENT (DR14)'    : df_exod_cat_dr14_transient,
           'EXOD & NOT TRANSIENT (DR14)': df_exod_cat_dr14_not_transient
           }

    return dfs

def main():
    savedir = data_simbad_stats 
    df_otypedef  = pd.read_csv(data_util / 'otypedef.csv')
    dfs = load_dfs()
    
    all_res = []
    all_res2 = []
    for k, df in dfs.items():
        # Use the simbad classifier to obtain the subclass of each otype
        df['simbad_subclass'] = df['otype'].apply(lambda x: simbad_classifier[x])

        # Convert all the otypes to the longnames
        df['otype'] = df['otype'].apply(lambda x: df_otypedef[df_otypedef['otype'] == x]['label'].values[0] if x in df_otypedef['otype'].values else x)
   
        # Add the description of each otype
        df['description'] = df['otype'].apply(lambda x: df_otypedef[df_otypedef['label'] == x]['description'].values[0] if x in df_otypedef['label'].values else x)
    
        n_total    = df.sumtotal
        n_objects  = df['object_count'].sum()
        n_missing  = n_total - n_objects
        n_otypes   = df['otype'].nunique()
        n_subtypes = df['simbad_subclass'].nunique()

        # Add the missing sources.
        df_missing = pd.DataFrame([{'otype'   : 'NO CMATCH',
                                    'object_count'    : n_missing,
                                    'percentage'      : n_missing / n_total * 100,
                                    'simbad_subclass' : 'NO CMATCH',
                                    'description'     : 'NO CMATCH'}])
        df = pd.concat([df, df_missing], ignore_index=True)
        df = df.sort_values('object_count', ascending=False).reset_index(drop=True)


        print('OTYPE statistics:')
        print(f'===============')
        print(f'Data Subset                : {k}')
        print(f'Number of objects (total)  : {n_total:,}')
        print(f'Number of objects (subset) : {n_objects:,}')
        print(f'Number of missing objects  : {n_missing:,}')
        print(f'Number of otypes           : {n_otypes:,}')
        print(f'Number of subtypes         : {n_subtypes:,}')
    
        print(f'otypes by object count:')
        print(f'=======================')
        print(df[['otype', 'simbad_subclass', 'description', 'object_count',  'percentage']])
        savepath = savedir / f'{k}_otype_stats.csv'
        print(f'Saving to {savepath}')
        df.to_csv(savepath, index=False)
    
    
        simbad_subclass_counts = df['simbad_subclass'].value_counts()
        res = {'data_subset': k,
               'n_total'   : n_total,
               'n_objects' : n_objects,
               'n_missing' : n_missing,
               'n_otypes'  : n_otypes,
               }
    
        # Create dictionary to store the number of objects for each otype
        for i, r in df.iterrows():
            d = {'data_subset' : k, 
                 'otype'       : r['otype'],
                 'subtype'     : r['simbad_subclass'],
                 'object_count': r['object_count'],
                 'percentage'  : r['percentage'],
                 'subset_count': n_total}
            all_res2.append(d)
    
        all_res.append(res)
        print('\n\n')
    
    df_res = pd.DataFrame(all_res)
    df_res['percentage'] = df_res['n_objects'] / df_res['n_total'] * 100
    df_res['percentage'] = df_res['percentage'].round(2)
    
    
    df_res2 = pd.DataFrame(all_res2)
    
    print('Number of objects in each data subset:')
    print('=======================================')
    print(df_res)
    print('\n\n')
    
    print('Number of objects for each otype in each data subset:')
    print('======================================================')
    print(df_res2)
    print('\n\n')
    
    # Calculate the percentage of each subtype in each data subset
    grouped                      = df_res2.groupby(['data_subset', 'subtype'])['object_count'].sum().reset_index()
    grouped['data_subset_count'] = grouped['data_subset'].map(df_res.set_index('data_subset')['n_objects'])
    grouped['total_count']       = grouped['data_subset'].map(df_res.set_index('data_subset')['n_total'])
    grouped['perc_ds']           = grouped['object_count'] / grouped['data_subset_count'] * 100
    grouped['perc_tot']          = grouped['object_count'] / grouped['total_count'] * 100
    grouped['perc_ds']           = grouped['perc_ds'].round(2)
    grouped['perc_tot']          = grouped['perc_tot'].round(2)
    
    print('Number of each simbad subtype in each data subset:')
    print('==================================================')
    print(grouped)
    savepath = savedir / 'n_subtypes_vs_data_subset.csv'
    print(f'Saving to {savepath}')
    grouped.to_csv(savepath, index=False)
    print('\n\n')
   
    pivot_table = grouped.pivot(index='subtype', columns='data_subset', values='object_count')
    print('Number of each simbad subtypes in each data subset:')
    print('===================================================')
    print(pivot_table)
    savepath = savedir / 'n_subtypes_vs_data_subset_pivot.csv'
    print(f'Saving to {savepath}')
    pivot_table.to_csv(savepath)
    print('\n\n')
    
    df_perc  = df_res2.pivot(index='otype', columns='data_subset', values='percentage')
    df_count = df_res2.pivot(index='otype', columns='data_subset', values='object_count')
    df_perc.reset_index(inplace=True)
    df_count.reset_index(inplace=True)
    
    print('Number of original (total) and filtered (n_objects) for each data subset:')
    print('=========================================================================')
    print(df_res)
    
    print('\n\n')
    print('OTYPE statistics (percentage):')
    print('==============================')
    print(df_perc.sort_values('SIMBAD', ascending=False))
    df_perc.to_csv(savedir / 'otype_stats_perc.csv', index=False)
    print(f'Saved to {savedir / "otype_stats_perc.csv"}')
    print('\n\n')
    
    print('OTYPE statistics (object count):')
    print('=================================')
    print(df_count.sort_values('SIMBAD', ascending=False))
    df_count.to_csv(savedir / 'otype_stats_count.csv', index=False)
    print(f'Saved to {savedir / "otype_stats_count.csv"}')


    dfs = read_simbad_stats()
    return dfs


def read_simbad_stats():
    all_savepaths = get_all_savepaths(load_dfs())
    dfs = {savepath: pd.read_csv(savepath) for savepath in all_savepaths}
    return dfs

if __name__ == '__main__':
    dfs = main()

   
