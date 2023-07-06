import optuna
import torch
import torch.nn as nn
import numpy as np
import h5py
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

class ModelBaseline(nn.Module):
    def __init__(self,):
        super(ModelBaseline, self).__init__()
        self.kernel_size = 3

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
                             out_features=1)
        
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


def train_loop(prefix, dataloader, model, optimizer, loss_function, device):
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
        loss = loss_function(nn.functional.sigmoid(output), diagnoses) # compute loss
        loss.backward() # compute gradients
        optimizer.step() # update parameters

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
        pred = torch.sigmoid(output) # apply sigmoid to get probabilities
        valid_pred.append(pred.cpu().numpy())
        valid_true.append(diagnoses_cpu.numpy())
        loss = loss_function(nn.functional.sigmoid(output), diagnoses) # compute loss
        
        # Update accumulated values
        total_loss += loss.detach().cpu().numpy()
        n_entries += len(traces)

    return total_loss / n_entries, np.vstack(valid_pred), np.vstack(valid_true)

def fit(learning_rate, weight_decay, num_epochs, model, train_dataloader, valid_dataloader ,seed=42, verbose=True, device="cuda:0"):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # =============== Define model ============================================#
    if verbose:
        print("Define model...")
    """
    TASK: Replace the baseline model with your model; Insert your code here
    """
    model = model()
    model.to(device=device)
    if verbose:
        print("Done!\n")

    # =============== Define loss function ====================================#
    """
    TASK: define the loss; Insert your code here. This can be done in 1 line of code
    """
    loss_function = torch.nn.BCELoss()

    # =============== Define optimizer ========================================#
    if verbose:
        print("Define optimiser...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    if verbose:
        print("Done!\n")

    # =============== Define lr scheduler =====================================#
    # TODO advanced students (non mandatory)
    """
    OPTIONAL: define a learning rate scheduler; Insert your code here
    """
    lr_scheduler = None

    # =============== Train model =============================================#
    if verbose:
        print("Training...")
    best_loss = np.Inf
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
        train_loss = train_loop(prefix, train_dataloader, model, optimizer, loss_function, device)
        # validation loop
        valid_loss, y_pred, y_true = eval_loop(prefix, valid_dataloader, model, loss_function, device)

        # collect losses
        train_loss_all.append(train_loss)
        valid_loss_all.append(valid_loss)

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
    
        # save best model: here we save the model only for the lowest validation loss
        if valid_loss < best_loss:
            # Save model parameters
            # torch.save({'model': model.state_dict()}, 'model.pth') 
            # Update best validation loss
            best_valid = valid_loss_all[-1]
            best_auroc = auroc_all[-1]
            best_accuracy = accuracy_all[-1]
            best_precision = precision_all[-1]
            best_recall = recall_all[-1]
            best_f1 = f1_all[-1]
            harmonic = 5 * (best_auroc * best_accuracy * best_precision * best_recall * best_f1)/(best_auroc + best_accuracy + best_precision + best_recall + best_f1 + 1e-6)
            best_loss = best_valid + (1-harmonic)
            # statement
            model_save_state = "Best model -> saved"
        else:
            model_save_state = ""

        if verbose and (epoch % 100 == 0 or model_save_state != ""):
            # Print message
            print('\rEpoch {epoch:2d}: \t'
                        'Train Loss {train_loss:.6f} \t'
                        'Valid Loss {valid_loss:.6f} \t'
                        'Auroc {auroc:.6f} \t'
                        '{model_save}'
                        .format(epoch=epoch,
                                train_loss=train_loss,
                                valid_loss=valid_loss,
                                auroc=auroc_all[-1],
                                model_save=model_save_state)
                            , end="")

        # Update learning rate with lr-scheduler
        if lr_scheduler:
            lr_scheduler.step()

    return best_loss, best_valid, best_auroc, best_accuracy, best_precision, best_recall, best_f1


def ask_tell_optuna(objective_func, study_name, storage_name):
    study = optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize", "maximize"], study_name=study_name, storage=storage_name, load_if_exists=True,)
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
        device,       
        trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-8, 1e2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e2, log=True)
    # print(f"Learning Rate {learning_rate}, Weight Decay {weight_decay}")
    best_loss, best_valid, best_auroc, best_accuracy, best_precision, best_recall, best_f1 = fit(learning_rate=learning_rate, 
                                                                                                     weight_decay=weight_decay, 
                                                                                                     num_epochs=num_epochs, 
                                                                                                     model=ModelBaseline, 
                                                                                                     train_dataloader=train_dataloader, 
                                                                                                     valid_dataloader=valid_dataloader,
                                                                                                     verbose=False, 
                                                                                                     device=device)
    return best_loss, best_valid, best_auroc, best_accuracy, best_f1