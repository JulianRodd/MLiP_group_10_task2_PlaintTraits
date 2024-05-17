from torch.optim import lr_scheduler
import torch
from pandas import DataFrame
import torchmetrics
import time
from kaggle_secrets import UserSecretsClient
import wandb
from datetime import date
from copy import deepcopy

from generics import Generics


def get_lr_scheduler(optimizer, config):
    """
    Takes:
        initialised optimizer
        config class instance
    Returns:
        OneCycleLR scheduler
    """
    return lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.LR_MAX,
        total_steps=config.N_STEPS["train"],
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=1e1,
        final_div_factor=1e1,
    )


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.sum += val.sum()
        self.count += val.numel()
        self.avg = self.sum / self.count


def r2_loss(y_pred, y_true, global_y_mean, eps=1e-6, device='cuda'):
    eps = torch.tensor([eps]).to(device)

    ss_res = torch.sum((y_true - y_pred) ** 2, dim=0)
    ss_total = torch.sum((y_true - global_y_mean) ** 2, dim=0)
    ss_total = torch.maximum(ss_total, eps)
    r2 = torch.mean(ss_res / ss_total)
    return r2


def get_y_mean(df: DataFrame):
    """
    Takes:
        dataframe like train
    Returns:
        Tensor of target columns means
    """
    return torch.tensor(df[Generics.TARGET_COLUMNS].values).mean(dim=0)


def init_wandb(
    config,
    secret_name="wandb_api",
    project_name="MLiP_PlantTraits",
    name="UnnamedRun",
):
    today = str(date.today())

    user_secrets = UserSecretsClient()

    wandb_api = user_secrets.get_secret(secret_name)

    wandb.login(key=wandb_api)

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project=project_name,
        name=f"{name}_{today[5:]}",
        # track hyperparameters and run metadata
        config=dict((name, getattr(config, name)) for name in dir(config) if not name.startswith('__'))
    )


def train(
    model,
    optimizer,
    config,
    scheduler,
    dataloader_train,
    dataloader_val = None,
    global_y_mean = None,
    loss_fn=r2_loss,
    use_wandb=True,
    wandb_kwargs: dict = None,
    save_checkpoint_every_epoch = True
    ):
    if dataloader_val is None:
        print("No validation set was given, so the best checkpoint will be based on the train R2")

    if use_wandb:
        if wandb_kwargs is not None:
            init_wandb(config, **wandb_kwargs)
        else:
            init_wandb(config)

    MAE = torchmetrics.regression.MeanAbsoluteError().to(config.DEVICE)
    R2 = torchmetrics.regression.R2Score(
        num_outputs=config.N_TARGETS, multioutput="uniform_average"
    ).to(config.DEVICE)

    LOSS = AverageMeter()
    best_val_r2 = 0
    model.train()
    for epoch in range(config.N_EPOCHS):
        model, scheduler, optimizer, train_r2, best_val_r2 = train_epoch(
            MAE,
            R2,
            LOSS,
            model,
            dataloader_train,
            dataloader_val,
            loss_fn,
            optimizer,
            scheduler,
            config,
            epoch,
            global_y_mean,
            best_val_r2,
            epoch,
            use_wandb,
            save_checkpoint_every_epoch
        )
        if use_wandb:
            wandb.log({"train_r2": train_r2})

        if dataloader_val is None:
            best_model_wts = deepcopy(model.state_dict())
            checkpoint = {
                'epoch': epoch,
                'model': best_model_wts,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': scheduler.state_dict()}
            if save_checkpoint_every_epoch:
                torch.save(checkpoint, f"{config.checkpoint_save_dir}best_model_epoch_{epoch}.pth")
            else:
                torch.save(checkpoint, f"{config.checkpoint_save_dir}best_model.pth")
            print("Saved checkpoint")

    if use_wandb:
        wandb.finish()
    torch.save(model.state_dict(), f"{config.checkpoint_save_dir}final_model.pth")
    return model


def train_epoch(
    MAE,
    R2,
    LOSS,
    model,
    dataloader,
    dataloader_val = None,
    loss_fn = None,
    optimizer = None,
    scheduler = None,
    config = None,
    current_epoch = None,
    global_y_mean = None,
    best_val_r2 = None,
    epoch = None,
    use_wandb=True,
    save_checkpoint_every_epoch=True
):

    MAE.reset()
    R2.reset()
    LOSS.reset()
    model.train()

    wandb.log({"epoch": epoch})

    for step, (X_batch, y_true) in enumerate(dataloader):
        wandb.log({"steps": step})
        X_batch = X_batch.to(config.DEVICE)
        y_true = y_true.to(config.DEVICE)
        t_start = time.perf_counter_ns()
        with torch.set_grad_enabled(True):
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_true, global_y_mean=global_y_mean.to(config.DEVICE), device=config.DEVICE)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        LOSS.update(loss)
        MAE.update(y_pred, y_true)
        R2.update(y_pred, y_true)

        batch_r2 = torchmetrics.functional.regression.r2_score(
            preds=y_pred, target=y_true
        )
        batch_mae = torchmetrics.functional.regression.mean_absolute_error(
            preds=y_pred, target=y_true
        )

        logging(
            config,
            "train",
            current_epoch,
            step,
            t_start,
            MAE,
            LOSS,
            R2,
            scheduler=scheduler,
            batch_mae=batch_mae,
            batch_r2=batch_r2,
            batch_loss=loss,
            use_wandb=use_wandb,
        )

        if dataloader_val is not None and step % config.VAL_STEPS == 0:
            model.eval()
            val_r2 = val_epoch(dataloader_val, config, global_y_mean, model, loss_fn)
            
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model_wts = deepcopy(model.state_dict())
                checkpoint = {
                    'epoch': epoch,
                    'model': best_model_wts,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': scheduler.state_dict()}
                if save_checkpoint_every_epoch:
                    torch.save(checkpoint, f"{config.checkpoint_save_dir}best_model_epoch_{epoch}.pth")
                else:
                    torch.save(checkpoint, f"{config.checkpoint_save_dir}best_model.pth")
                print("Saved checkpoint")
                
            model.train()

    return model, scheduler, optimizer, R2.compute().item(), best_val_r2


def val_epoch(dataloader_val, config, global_y_mean, model, loss_fn):
    running_val_loss = 0.0
    with torch.set_grad_enabled(False):
        y_true = []
        y_pred = []

        for inputs_val, labels_val in dataloader_val:
            inputs_val = inputs_val.to(config.DEVICE)
            labels_val = labels_val.to(config.DEVICE)
            outputs_val = model(inputs_val)
            val_loss = loss_fn(outputs_val, labels_val, global_y_mean, device=config.DEVICE)
            running_val_loss += val_loss.item() * inputs_val.size(0)
            y_true.append(labels_val.to("cpu"))
            y_pred.append(outputs_val.to("cpu"))

        y_true = torch.cat(y_true)
        y_pred = torch.cat(y_pred)
            
        val_loss = running_val_loss / len(dataloader_val.dataset)
        wandb.log({"val_r2_loss": val_loss})
        print('Val loss: {:.4f}'.format(val_loss))

        val_r2 = torchmetrics.functional.regression.r2_score(preds=y_pred, target=y_true)
        wandb.log({"val_r2": val_r2})
        print('Val R2: {:.4f}'.format(val_r2))


        for i, target_feat in enumerate(Generics.TARGET_COLUMNS):
            val_r2_feat = torchmetrics.functional.regression.r2_score(preds=y_pred[:, i], target=y_true[:, i])
            wandb.log({f"val_r2_{target_feat}": val_r2_feat})

        return val_r2


def logging(
    config,
    mode,
    epoch,
    step,
    t_start,
    MAE,
    LOSS,
    R2,
    batch_loss,
    batch_r2,
    batch_mae,
    use_wandb=True,
    scheduler=None,
):
    if not config.IS_INTERACTIVE and (step + 1) == config.N_STEPS_PER_EPOCH[mode]:
        print(
            get_log_string(config, mode, epoch, step, t_start, MAE, LOSS, R2, scheduler)
        )
    elif config.IS_INTERACTIVE:
        print(
            get_log_string(
                config, mode, epoch, step, t_start, MAE, LOSS, R2, scheduler
            ),
            end="\n" if (step + 1) == config.N_STEPS_PER_EPOCH[mode] else "",
            flush=True,
        )
    if use_wandb:
        logging_dict = {
            f"{mode}_r2_batch_loss": batch_loss.item(),
            f"{mode}_r2_batch": batch_r2.item(),
            f"{mode}_mae_batch": batch_mae.item(),
        }
        if scheduler is not None:
            logging_dict["lr"] = scheduler.get_last_lr()[0]

        wandb.log(logging_dict)


def get_log_string(config, mode, epoch, step, t_start, MAE, LOSS, R2, scheduler=None):
    string = (
        f"\rEPOCH[{mode}] {epoch+1:02d}, {step+1:04d}/{config.N_STEPS_PER_EPOCH[mode]} | "
        + f"loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, "
    )

    if mode == "train":
        string += f"step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {scheduler.get_last_lr()[0]:.2e}"
    else:
        string += f"step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s"
    return string
