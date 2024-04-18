import pandas as pd 
from numpy import log10

from generics import Generics


def outlier_filter(df:pd.DataFrame, lower_quantile=0.005, upper_quantile=0.985):
    '''
    Filters target columns

    Takes: 
        pd.DataFrame with target columns

    Returns: 
        filtered dataframe
    '''
    for column in Generics.TARGET_COLUMNS:
        lower_quantile = df[column].quantile(lower_quantile)
        upper_quantile = df[column].quantile(upper_quantile)  
        df = df[(df[column] >= lower_quantile) & (df[column] <= upper_quantile)]
        return df 
    
def log_transform(df:pd.DataFrame, columns=None):
    '''
    Log transforms columns 
    If None given applies it to all target columns 

    Takes:
        train DataFrame
    Returns:
        DataFrame with log10 transformed cols 
    '''
    for target_idx, target in enumerate(Generics.TARGET_COLUMNS):
        v = df[target].values
        if target in columns:
            v = log10(v)
        df[:, target_idx] = v
    return df 

