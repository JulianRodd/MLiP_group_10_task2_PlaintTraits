import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from generics import Generics
from utils.preprocessing_utils import log_transform, outlier_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directories(config):
    os.makedirs(config.checkpoint_save_dir, exist_ok=True)
    os.makedirs(config.model_save_dir, exist_ok=True)


def prep_dataset(
    filepath, size=None, train_size=0.8, seed=42, tabular_only=False, vit_features=None
):
    df = pd.read_csv(filepath)
    if vit_features is not None:
        vit_features = pd.read_csv(vit_features)
        df = pd.concat([df, vit_features], axis=1)
    if size is not None:
        df = df.sample(size, random_state=seed)

    if not tabular_only:
        df["file_path"] = df["id"].apply(
            lambda s: f"/kaggle/input/planttraits2024/train_images/{s}.jpeg"
        )
        df["jpeg_bytes"] = df["file_path"].progress_apply(
            lambda fp: open(fp, "rb").read()
        )

    df = df[df["X4_mean"] > 0]

    if train_size is None:
        shuffled_df = df.sample(n=len(df), random_state=seed)
        shuffled_df = shuffled_df.reset_index(drop=True)
        return shuffled_df, None

    else:
        train, val = train_test_split(df, train_size=train_size, random_state=seed)
        return train, val


def load_data(
    train_path,
    test_path,
    subset_size=None,
    vit_features_train=None,
    vit_features_test=None,
):
    test_df = pd.read_csv(test_path)
    if vit_features_test is not None:
        vit_features_test = pd.read_csv(vit_features_test)
        test_df = pd.concat([test_df, vit_features_test], axis=1)

    train_df, val_df = prep_dataset(
        train_path, train_size=0.9, tabular_only=True, vit_features=vit_features_train
    )
    if subset_size:
        train_df = train_df.sample(n=subset_size, random_state=42)
        val_df = val_df.sample(n=subset_size // 4, random_state=42)
        test_df = test_df.sample(n=subset_size // 4, random_state=42)
    train_df = outlier_filter(train_df)
    val_df = outlier_filter(val_df)
    return train_df, val_df, test_df


def set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def scale_data(train_df, val_df, test_df, columns):
    scaler = StandardScaler()
    for col in columns:
        train_df[col] = scaler.fit_transform(train_df[col].values.reshape(-1, 1))
        val_df[col] = scaler.transform(val_df[col].values.reshape(-1, 1))
        test_df[col] = scaler.transform(test_df[col].values.reshape(-1, 1))
    return train_df, val_df, test_df, scaler


def log_transform_data(df, columns):
    return log_transform(df, columns=columns)


def apply_pca(train_df, val_df, test_df, n_components=25):
    pca = PCA(n_components=n_components)
    train_df_features = pca.fit_transform(train_df)
    val_df_features = pca.transform(val_df)
    test_df_features = pca.transform(test_df)
    return train_df_features, val_df_features, test_df_features, pca


def combine_features_with_targets(features, targets, columns_prefix="PC"):
    features_df = pd.DataFrame(
        features, columns=[f"{columns_prefix}{i}" for i in range(features.shape[1])]
    )
    combined_df = pd.concat(
        [features_df.reset_index(drop=True), targets.reset_index(drop=True)], axis=1
    )
    return combined_df


def train_model(
    train_df, val_df, config, model_config, trainer_config, model_name, target
):
    model_target_dict = {}
    model_path = os.path.join(config.model_save_dir, f"{model_name}_{target}.pth")
    if config.use_cached_models and os.path.exists(model_path):
        logger.info(f"Loading cached model for {target} from {model_path}")
        tabular_model = TabularModel.load_from_checkpoint(model_path)
    else:
        data_config = DataConfig(
            target=[target],
            continuous_cols=config.TABULAR_COLUMNS,
        )
        optimizer_config = OptimizerConfig()
        tabular_model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
        )
        tabular_model.fit(train=train_df, validation=val_df)
        logger.info(f"Saved model for {target} to {model_path}")
    model_target_dict[target] = tabular_model
    return model_target_dict


def predict_single(model_target_dict, x, config):
    y_pred = np.zeros((1, config.N_TARGETS))
    x_df = pd.DataFrame([x], columns=config.TABULAR_COLUMNS)
    for i, target in enumerate(Generics.TARGET_COLUMNS):
        if target in model_target_dict:
            model = model_target_dict[target]
            y_pred[:, i] = model.predict(x_df).values
    return y_pred


def predict_batch(model_target_dict, df, config):
    predictions = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        x_sample = row[config.TABULAR_COLUMNS].values
        y_pred = predict_single(model_target_dict, x_sample, config)
        predictions.append(y_pred)
    return predictions


def generate_submission(test_df, predictions, config, SCALER, filename):
    submission_rows = []
    for y_pred, test_id in zip(predictions, test_df["id"]):
        row = {"id": test_id}
        inverse_pred = SCALER.inverse_transform(y_pred)
        for k, v in zip(config.TARGET_COLUMNS, inverse_pred[0]):
            if k in config.LOG_FEATURES:
                row[k.replace("_mean", "")] = 10**v
            else:
                row[k.replace("_mean", "")] = v
        submission_rows.append(row)
    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv(filename, index=False)
    print(f"Submission saved to {filename}!")


def non_pytorch_r2_loss(y_true, y_pred, global_y_mean, eps=1e-6):
    eps = np.array([eps])
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_total = np.sum((y_true - global_y_mean) ** 2, axis=0)
    ss_total = np.maximum(ss_total, eps)
    r2 = np.mean(ss_res / ss_total)
    return r2


def evaluate_model_on_val(model_target_dict, val_df, config, model_name):
    y_true = val_df[config.TARGET_COLUMNS].values
    y_pred = np.zeros_like(y_true)
    for i, row in enumerate(tqdm(val_df[config.TABULAR_COLUMNS].values)):
        y_pred[i] = predict_single(model_target_dict, row, config)

    if np.isnan(y_pred).any():
        logger.warning(
            f"NaN values found in predictions for model {model_name}. Imputing with mean."
        )
        y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred))
    global_y_mean = np.mean(y_true, axis=0)
    r2 = non_pytorch_r2_loss(y_true, y_pred, global_y_mean)
    results = {"model": model_name, "Overall R2": r2}

    for i, target in enumerate(config.TARGET_COLUMNS):
        r2_individual = r2_score(y_true[:, i], y_pred[:, i])
        results[f"R2 for {target}"] = r2_individual

    return results


def ensemble_predictions(models_predictions, config):
    ensembled_predictions = np.mean(models_predictions, axis=0)
    return ensembled_predictions


def generate_ensemble_submission(test_df, predictions_dict, config, SCALER):
    ensembled_rows = []

    for idx, test_id in enumerate(test_df["id"]):
        row = {"id": test_id}
        for target in config.TARGET_COLUMNS:
            models = config.MODEL_ENSEMBLE_DICT[target]
            target_preds = []
            for model_name in models:
                model_predictions = pd.read_csv(config.MODEL_CSV_DICT[model_name])
                target_preds.append(
                    model_predictions.loc[
                        model_predictions["id"] == test_id, target.replace("_mean", "")
                    ].values[0]
                )
            ensembled_pred = np.mean(target_preds)
            if target in config.LOG_FEATURES:
                row[target.replace("_mean", "")] = 10**ensembled_pred
            else:
                row[target.replace("_mean", "")] = ensembled_pred
        ensembled_rows.append(row)

    ensemble_df = pd.DataFrame(ensembled_rows)
    ensemble_df.to_csv(config.MODEL_CSV_DICT["ensemble"], index=False)
    print("Ensemble submission saved to ensemble_submission.csv")


def plot_heatmap_and_correlation(data, title, output_path):
    corr = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        xticklabels=False,
        yticklabels=False,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
