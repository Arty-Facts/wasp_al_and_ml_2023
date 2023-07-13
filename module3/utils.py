import optuna
import torch
import torch.nn as nn
import numpy as np
import h5py, os
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from sklearn.metrics import roc_auc_score
from tqdm.notebook import trange, tqdm


import optuna
import time
from functools import partial
from collections import defaultdict

import multiprocessing

from pathlib import Path

from models import Baseline, Model, Model_V2

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_labels(x, threshold=0.5):
    return (x > threshold).astype(np.float32)

def accuracy_score(y_true, y_pred):
    """
    Computes the accuracy score.

    """
    return np.mean(y_true == to_labels(y_pred))

def precision_score(y_true, y_pred, eps=1e-6):
    """
    Computes the precision score.

    Precision = TP / (TP + FP)
    """
    y_pred = to_labels(y_pred)
    return np.sum(y_true * y_pred) / (np.sum(y_pred) + eps)

def recall_score(y_true, y_pred, eps=1e-6):
    """
    Computes the recall score.

    Recall = TP / (TP + FN)
    """
    y_pred = to_labels(y_pred)
    return np.sum(y_true * y_pred) / (np.sum(y_true) + eps)

def f1_score(y_true, y_pred, eps=1e-6):
    """
    Computes the F1 score.

    F1 = 2 * (precision * recall) / (precision + recall)
    """
    y_pred = to_labels(y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall+eps)

class Baseline(nn.Module):
    def __init__(self, out_features=1, name="palceholder",  training_args={}):
        super(Baseline, self).__init__()
        self.kernel_size = 3
        self.name = f'Baseline_F{out_features}'
        self.kvargs = {
            'name': self.name,
            'training_args': training_args,
            'out_features': out_features
        }
        self.out_features = out_features

        # conv layer
        downsample = self._downsample(4096, 128)
        self.conv1 = nn.Conv1d(in_channels=8, 
                               out_channels=32, 
                               kernel_size=self.kernel_size, 
                               stride=downsample,
                               padding=self._padding(downsample),
                               bias=False)
        
        # linear layer
        self.lin = nn.Linear(in_features=32*128,
                             out_features=out_features)
        
        # ReLU
        self.relu = nn.ReLU()

    def _padding(self, downsample):
        return max(0, int(np.floor((self.kernel_size - downsample + 1) / 2)))

    def _downsample(self, seq_len_in, seq_len_out):
        return int(seq_len_in // seq_len_out)


    def forward(self, x):
        x= x.transpose(2,1)

        x = self.relu(self.conv1(x))
        x_flat= x.view(x.size(0), -1)
        x = self.lin(x_flat)

        return x


def train_loop(prefix, dataloader, model, optimizer, lr_scheduler, loss_function, device):
    # model to training mode (important to correctly handle dropout or batchnorm layers)
    model.train()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    # training loop
    for it, (traces, diagnoses) in enumerate(dataloader):
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces.to(device), diagnoses.to(device)

        optimizer.zero_grad()  # set gradients to zero
        output = model(traces) # forward pass
        loss = loss_function(output, diagnoses) # compute loss
        loss.backward() # compute gradients
        optimizer.step() # update parameters
        if lr_scheduler:
            lr_scheduler.step()


        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_entries += len(traces)

    return total_loss / n_entries

def eval_loop(prefix, dataloader, model, loss_function, device):
    # model to evaluation mode (important to correctly handle dropout or batchnorm layers)
    model.eval()
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    valid_pred, valid_true = [], []
    # evaluation loop
    for it, (traces_cpu, diagnoses_cpu) in enumerate(dataloader):
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)

        with torch.no_grad(): # no gradients needed
            output = model(traces) # forward pass

        if output.shape[1] > 1:
            pred = torch.sigmoid(output[:,0]).unsqueeze(1)
            diagnoses_cpu = diagnoses_cpu[:,0].unsqueeze(1)
        else:
            pred = torch.sigmoid(output)
        valid_pred.append(pred.cpu().numpy())
        valid_true.append(diagnoses_cpu.numpy())
        loss = loss_function(output, diagnoses) # compute loss
        
        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_entries += len(traces)

    return total_loss / n_entries, np.vstack(valid_pred), np.vstack(valid_true)

def binary_cross_entropy(output, target):
    return torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), target, reduction='mean')


def fit(num_epochs, model, optimizer, train_dataloader, valid_dataloader, loss_function=binary_cross_entropy, lr_scheduler=None ,seed=42, verbose=True, device="cuda:0", trial=-1, prefix=""):
    np.random.seed(seed)
    torch.manual_seed(seed)

    model.to(device=device)

    if not Path(f'saved_models{prefix}').exists():
        Path(f'saved_models{prefix}').mkdir()
    filename = f'saved_models{prefix}/{model.name}_{device.replace(":", "_")}.pth'

    # =============== Train model =============================================#
    if verbose:
        print("Training...")
    best_score = np.Inf

    best_global = None
    if os.path.exists(filename):
        ckpt = torch.load(filename)
        best_global = ckpt['score']

    # allocation
    train_loss_all, valid_loss_all, auroc_all = [], [], []
    accuracy_all, precision_all, recall_all, f1_all = [], [], [], []
    if verbose:
        pbar = tqdm(range(1, num_epochs + 1))
    else:
        pbar = range(1, num_epochs + 1)
    # loop over epochs
    for epoch in pbar:
        # update progress bar set prefix f"epoch {epoch}/{num_epochs}"
        prefix = f""
        
        # training loop
        train_loss = train_loop(prefix, train_dataloader, model, optimizer,lr_scheduler, loss_function, device)
        # validation loop
        valid_loss, y_pred, y_true = eval_loop(prefix, valid_dataloader, model, loss_function, device)
        # collect losses
        train_loss_all.append(train_loss)
        valid_loss_all.append(valid_loss)

        # if v3 then the labels are different select only the first column

        if y_pred.shape[1] > 1:
            y_pred = y_pred[:, 0]
        if y_true.shape[1] > 1:
            y_true = y_true[:, 0]

        # compute validation metrics for performance evaluation
        """
        TASK: compute validation metrics (e.g. AUROC); Insert your code here
        This can be done e.g. in 5 lines of code
        """
        auroc_all.append(roc_auc_score(y_true, y_pred))
        accuracy_all.append(accuracy_score(y_true, y_pred))
        precision_all.append(precision_score(y_true, y_pred))
        recall_all.append(recall_score(y_true, y_pred))
        f1_all.append(f1_score(y_true, y_pred))
        curr_valid = valid_loss_all[-1]
        curr_auroc = auroc_all[-1]
        curr_accuracy = accuracy_all[-1]
        curr_precision = precision_all[-1]
        curr_recall = recall_all[-1]
        curr_f1 = f1_all[-1]
        harmonic = 5 * (curr_auroc * curr_accuracy * curr_precision * curr_recall * curr_f1)/(curr_auroc + curr_accuracy + curr_precision + curr_recall + curr_f1 + 1e-6)
        curr_loss = curr_valid + (1-harmonic)

        # save best model: here we save the model only for the lowest validation loss
        if curr_loss < best_score:
            # Save model parameters
            # torch.save({'model': model.state_dict()}, 'model.pth') 
            # Update best validation loss
            best_valid = valid_loss_all[-1]
            best_auroc = auroc_all[-1]
            best_accuracy = accuracy_all[-1]
            best_precision = precision_all[-1]
            best_recall = recall_all[-1]
            best_f1 = f1_all[-1]
            best_score = curr_loss
            # statement
            # save model
            if best_global is None or best_score < best_global:
                torch.save({'model': model.state_dict(), 
                            "trial": trial,
                            'kvargs': model.kvargs,
                            'num_epochs': num_epochs,
                            'score': best_score, 
                            'valid': best_valid, 
                            'auroc': best_auroc, 
                            'accuracy': best_accuracy, 
                            'precision': best_precision, 
                            'recall': best_recall, 
                            'f1': best_f1}, 
                            filename)
                best_global = best_score
        # Update learning rate with lr-scheduler
       
    return best_score, best_valid, best_auroc, best_accuracy, best_precision, best_recall, best_f1


def ask_tell_optuna(objective_func, study_name, storage_name):
    study = optuna.load_study(study_name=study_name, storage=storage_name)
    trial = study.ask()
    res = objective_func(trial)
    study.tell(trial, res)

def optimize(func, gpus, possesses, trials, study_name, storage_name):
    running_processes = defaultdict(list)
    trials_left = trials
    bar = tqdm(total=trials, desc="Trials", smoothing=0.001)
    while trials_left > 0:
        for gpu_index in range(gpus):
            device = f"cuda:{gpu_index}"
            objective = partial(func, device)
            
            while trials_left > 0 and len(running_processes[device]) < possesses:
                p = multiprocessing.Process(target=ask_tell_optuna, args=(objective, study_name, storage_name))
                p.start()
                time.sleep(1)
                running_processes[device].append(p)
                trials_left -= 1
                # print(f"Trials Left {trials_left}")
        for name, active_processes in running_processes.items():
            for process in active_processes:
                if not process.is_alive():
                    bar.update(1)
                    running_processes[name].remove(process)
        time.sleep(1)  # Adjust as needed to control the frequency of checking
        
    for name, active_processes in running_processes.items():
        for process in active_processes:
            process.join() # Wait for all processes to finish
            bar.update(1)
    bar.close()

def base_model_objective(
        num_epochs,
        train_dataloader,
        valid_dataloader,
        out_features, 
        prefix,
        device,       
        trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-9, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True)

    model=Baseline(
        out_features=out_features,
        training_args={
            "optimizer": {
                "name": "Adam",
                "lr": learning_rate,
                "weight_decay": weight_decay,
            },
            "lr_scheduler": {
                "name": "OneCycleLR",
                "max_lr": learning_rate,
                "steps_per_epoch": len(train_dataloader[out_features]),
                "epochs": num_epochs,
            },
            "batch_size": train_dataloader[out_features].batch_size,
            "num_epochs": num_epochs,
        }
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader[model.out_features]), epochs=num_epochs)

    # print(f"Learning Rate {learning_rate}, Weight Decay {weight_decay}")
    best_score, best_valid, best_auroc, best_accuracy, best_precision, best_recall, best_f1 = fit(
        num_epochs=num_epochs, 
        model=model, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader[model.out_features], 
        valid_dataloader=valid_dataloader[model.out_features],
        verbose=False, 
        device=device, 
        trial=trial.number,
        prefix=prefix
        )
    return best_score, best_valid, best_auroc, best_accuracy, best_f1

def model_objective(
        num_epochs,
        train_dataloader,
        valid_dataloader,
        out_features, 
        prefix,
        device,       
        trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-9, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-2, log=True)


    kernel_size= trial.suggest_int("kernel_size", 3, 65, step=2)
    steps = trial.suggest_int("steps", 0, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    num_layers = trial.suggest_int("num_layers", 1, 7)
    lin_steps = trial.suggest_int("lin_steps", 0, 3)

    model=Model(
            kernel_size=kernel_size, 
            num_layers=num_layers,
            steps=steps, 
            dropout=dropout, 
            lin_steps=lin_steps, 
            out_features=out_features,
            training_args={
                "optimizer": {
                    "name": "Adam",
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                },
                "lr_scheduler": {
                    "name": "OneCycleLR",
                    "max_lr": learning_rate,
                    "steps_per_epoch": len(train_dataloader[out_features]),
                    "epochs": num_epochs,
                },
                "batch_size": train_dataloader[out_features].batch_size,
                "num_epochs": num_epochs,
            }
        )
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader[model.out_features]), epochs=num_epochs)


    best_score, best_valid, best_auroc, best_accuracy, best_precision, best_recall, best_f1 = fit(
        num_epochs=num_epochs, 
        model=model, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader[model.out_features], 
        valid_dataloader=valid_dataloader[model.out_features],
        verbose=False, 
        device=device,
        trial=trial.number,
        prefix=prefix
        )
    p_count = count_parameters(model)
    return best_score, best_valid, best_auroc, best_accuracy, best_f1

def model_v2_objective(
        num_epochs,
        train_dataloader,
        valid_dataloader,
        out_features,
        prefix,
        device,       
        trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-9, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-9, 1e-3, log=True)

    kernel_size= trial.suggest_int("kernel_size", 3, 65, step=2)
    steps = trial.suggest_int("steps", 0, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)

    encode_layers = trial.suggest_int("encode_layers", 1, 9)
    encoder_out_channels = trial.suggest_int("encoder_out_channels", 32, 1024, step=32)

    reduce_layers = trial.suggest_int("reduce_layers", 1, 7)
    reduce_out_channels = trial.suggest_int("reduce_out_channels", 1, 32, step=1)

    lin_dims = trial.suggest_int("lin_dims", 32, 512, step=32)
    lin_steps = trial.suggest_int("lin_steps", 0, 3)

    model=Model_V2(
            kernel_size=kernel_size, 
            encode_layers=encode_layers,
            encoder_out_channels=encoder_out_channels,
            reduce_layers=reduce_layers,
            reduce_out_channels=reduce_out_channels,
            steps=steps, 
            dropout=dropout, 
            lin_steps=lin_steps, 
            lin_dims=lin_dims,
            out_features=out_features,
            training_args={
                "optimizer": {
                    "name": "Adam",
                    "lr": learning_rate,
                    "weight_decay": weight_decay,
                },
                "lr_scheduler": {
                    "name": "OneCycleLR",
                    "max_lr": learning_rate,
                    "steps_per_epoch": len(train_dataloader[out_features]),
                    "epochs": num_epochs,
                },
                "batch_size": train_dataloader[out_features].batch_size,
                "num_epochs": num_epochs,
            }
        )
    optimizer=torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dataloader[model.out_features]), epochs=num_epochs)


    best_score, best_valid, best_auroc, best_accuracy, best_precision, best_recall, best_f1 = fit(
        num_epochs=num_epochs, 
        model=model, 
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_dataloader=train_dataloader[model.out_features], 
        valid_dataloader=valid_dataloader[model.out_features],
        verbose=False, 
        device=device, 
        trial=trial.number,
        prefix=prefix
        )
    p_count = count_parameters(model)
    return best_score, best_valid, best_auroc, best_accuracy, best_f1



