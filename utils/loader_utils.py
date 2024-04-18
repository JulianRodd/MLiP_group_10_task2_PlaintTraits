import pandas as pd 
import numpy as np
from torch.utils.data import Dataset, DataLoader
import imageio

from generics import Generics


def prep_dataset(filepath):
    '''
    Takes:
      filepath to csv file 
    Returns: 
        pd.DataFrame with filepaths for images and jpeg bytes added
    '''
    df = pd.read_csv(filepath)
    df['file_path'] = df['id'].apply(lambda s: f'/kaggle/input/planttraits2024/train_images/{s}.jpeg')
    df['jpeg_bytes'] = df['file_path'].progress_apply(lambda fp: open(fp, 'rb').read())
    return df 


class Dataset(Dataset):
    '''
    Dataset class with jpeg_bytes as data and targets as y 
    '''
    def __init__(self, X_jpeg_bytes, y, transforms=None):
        self.X_jpeg_bytes = X_jpeg_bytes
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.X_jpeg_bytes)

    def __getitem__(self, index):
        X_sample = self.transforms(
            image=imageio.imread(self.X_jpeg_bytes[index]),
        )['image']
        y_sample = self.y[index]
        
        return X_sample, y_sample
    

def get_dataset(df:pd.DataFrame, transforms=None):
    '''
    Takes:
        df: a pd.DataFrame object as in train or test
        transforms: Albumentations.Compose object with transformations to be applied 
    Returns: 
        torch.Dataset object with images as data and target columns as targets
    '''

    y_vals = np.array(df[Generics.TARGET_COLUMNS].values)
    dataset = Dataset(
        df['jpeg_bytes'].values,
        y_vals,
        transforms
        )
    return dataset
    
