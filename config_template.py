import os
from generics import Generics


class Config:
    IMAGE_SIZE = 224
    N_TARGETS = len(Generics.TARGET_COLUMNS)
    DATASET_SIZE = None
    
    BATCH_SIZE = 8
    LR_MAX = 1e-5
    WEIGHT_DECAY = 0.01
    N_EPOCHS = 10
    VAL_STEPS = 500   # every n-th train step, a valdiation step is performed
    TRAIN_MODEL = True
    PRETRAINED = True
    IS_INTERACTIVE = os.environ['KAGGLE_KERNEL_RUN_TYPE'] == 'Interactive'

    DEVICE = "cuda"
    SEED = 42
    
#   Model
    MODEL = "vit_large_patch16"
    DROPOUT = 0.1
    GLOBAL_POOL = False
    FREEZE_BODY = False

    # Data
    train_csv_path = "/kaggle/input/planttraits2024/train.csv" 
    train_imgs_path = "/kaggle/input/planttraits2024/train_images" 
    test_csv_path = "/kaggle/input/planttraits2024/test.csv"
    test_imgs_path = "/kaggle/input/planttraits2024/test_images"
    log_dir = "/kaggle/working/log"
    checkpoint_save_dir = "/kaggle/working/saved_checkpoints/"
    checkpoint_path = "/kaggle/input/plantclef2022_mae_vit_large_patch16_epoch100/pytorch/plantclef2022_mae_vit_large_patch16_epoch100/1/PlantCLEF2022_MAE_vit_large_patch16_epoch100.pth"        
