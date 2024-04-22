from torch.optim import lr_scheduler
import torch 
from pandas import DataFrame
import torchmetrics
import time 

from generics import Generics


def get_lr_scheduler(optimizer, config):
    '''
    Takes: 
        initialised optimizer 
        config class instance 
    Returns: 
        OneCycleLR scheduler 
    '''
    return lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=config.LR_MAX,
        total_steps=config.N_STEPS,
        pct_start=0.1,
        anneal_strategy='cos',
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


def r2_loss(y_pred, y_true, global_y_mean, eps=1e-6):
    ss_res = torch.sum((y_true - y_pred)**2, dim=0)
    global_y_mean.to('cuda')
    ss_total = torch.sum((y_true - global_y_mean)**2, dim=0)
    eps = torch.tensor([eps]).to('cuda')
    ss_total = torch.maximum(ss_total, eps)
    r2 = torch.mean(ss_res / ss_total)
    return r2


def get_y_mean(df:DataFrame):
    '''
    Takes: 
        dataframe like train
    Returns:
        Tensor of target columns means  
    '''
    return torch.tensor(df[Generics.TARGET_COLUMNS]).mean(dim=0)


def train(model, optimizer, config, scheduler, dataloader, loss_fn=r2_loss):
    MAE = torchmetrics.regression.MeanAbsoluteError().to('cuda')
    R2 = torchmetrics.regression.R2Score(num_outputs=config.N_TARGETS, multioutput='uniform_average').to('cuda')
    LOSS = AverageMeter()

    for epoch in range(config.N_EPOCHS):
        MAE.reset()
        R2.reset()
        LOSS.reset()
        model.train()
            
        for step, (X_batch, y_true) in enumerate(dataloader):
            X_batch = X_batch.to('cuda')
            y_true = y_true.to('cuda')
            t_start = time.perf_counter_ns()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_true)
            LOSS.update(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            MAE.update(y_pred, y_true)
            R2.update(y_pred, y_true)
                
            if not config.IS_INTERACTIVE and (step+1) == config.N_STEPS_PER_EPOCH:
                print(
                    f'EPOCH {epoch+1:02d}, {step+1:04d}/{config.N_STEPS_PER_EPOCH} | ' + 
                    f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                    f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {scheduler.get_last_lr()[0]:.2e}',
                )
            elif config.IS_INTERACTIVE:
                print(
                    f'\rEPOCH {epoch+1:02d}, {step+1:04d}/{config.N_STEPS_PER_EPOCH} | ' + 
                    f'loss: {LOSS.avg:.4f}, mae: {MAE.compute().item():.4f}, r2: {R2.compute().item():.4f}, ' +
                    f'step: {(time.perf_counter_ns()-t_start)*1e-9:.3f}s, lr: {scheduler.get_last_lr()[0]:.2e}',
                    end='\n' if (step + 1) == config.N_STEPS_PER_EPOCH else '', flush=True,
                )

    torch.save(model, 'model.pth')
    return model 

