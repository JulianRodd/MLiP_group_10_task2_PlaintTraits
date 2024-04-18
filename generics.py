import numpy as np

class Generics: 
    TARGET_COLUMNS = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    MEAN_FILTERED = np.array([0.485, 0.456, 0.406]) # we assume that this is filtered due to the difference in our own vals 
    STD_FILTERED = np.array([0.229, 0.224, 0.225])
    MEAN_UNFILTERED = np.array([0.4446020961299334, 0.4495755761437776, 0.3355966340957563])
    STD_UNFILTERED = np.array([0.23482480079144202, 0.22604785671222852, 0.23545116111356945])


class Paths: 
    TRAIN_CSV = '/kaggle/input/planttraits2024/train.csv'
    TEST_CSV = '/kaggle/input/planttraits2024/test.csv'