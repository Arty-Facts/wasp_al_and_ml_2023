from utils import optimize, base_model_objective, model_objective, model_v2_objective
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import trange, tqdm
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms

from functools import partial

import multiprocessing
try:
    multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass
import glob, json
from pathlib import Path
import optuna
import random
import os
from torch.utils.data import TensorDataset, random_split, DataLoader, Dataset    
import h5py

PATH_TO_H5_FILE = 'codesubset/train.h5'
f = h5py.File(PATH_TO_H5_FILE, 'r')
data = f['tracings']


batch_size = 32


def to_labels(x, threshold=0.5):
    return (x > threshold).astype(np.float32)

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == to_labels(y_pred))

def precision_score(y_true, y_pred, eps=1e-6):
    """
    Precision = TP / (TP + FP)
    """
    y_pred = to_labels(y_pred)
    return np.sum(y_true * y_pred) / (np.sum(y_pred) + eps)

def recall_score(y_true, y_pred, eps=1e-6):
    """
    Recall = TP / (TP + FN)
    """
    y_pred = to_labels(y_pred)
    return np.sum(y_true * y_pred) / (np.sum(y_true) + eps)

def f1_score(y_true, y_pred, eps=1e-6):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    """
    y_pred = to_labels(y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * precision * recall / (precision + recall+eps)

def binary_cross_entropy(output, target):
    return torch.nn.functional.binary_cross_entropy(torch.sigmoid(output), target, reduction='mean')

def thee_ways_loss(output, target):
    loss_af = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output[:, 0]), target[:, 0], reduction='mean')
    loss_sex = torch.nn.functional.binary_cross_entropy(torch.sigmoid(output[:, 1]), target[:, 1],  reduction='mean')
    loss_age = torch.nn.functional.mse_loss(output[:, 2], target[:, 2], reduction='mean')
    return loss_af + loss_sex + loss_age

meta_data = pd.read_csv('codesubset/train.csv')
ids_traces = [int(x.split('TNMG')[1]) for x in list(pd.read_csv('codesubset/train/RECORDS.txt', header=None)[0])] # Get order of ids in traces
INDEX_TO_ID = meta_data["id_exam"]
ID_TO_INDEX = {id: i for i, id in enumerate(INDEX_TO_ID)}   # map id to index
meta_data.set_index('id_exam', inplace=True)
meta_data = meta_data.reindex(ids_traces) # make sure the order is the same
meta_data["date_exam"] = meta_data["date_exam"].astype("datetime64[ms]")

def split_data_by_age_sex_af(data, age_range=5, min_age=10,  max_age=105):
    # Calculate the lower and upper age bounds for each age range
    age_ranges = np.arange(min_age, max_age, age_range)

    # Create an additional 'age_range' column based on the age bounds
    data['age_range'] = pd.cut(data['age'], bins=list(age_ranges) + [max_age])
    male_sick = data[(data['sex'] == 'M') & (data["AF"] == 1)].groupby(['age_range']).size().reset_index()[0]
    male_clear = data[(data['sex'] == 'M') & (data["AF"] == 0)].groupby(['age_range']).size().reset_index()[0]

    female_sick = data[(data['sex'] == 'F') & (data["AF"] == 1)].groupby(['age_range']).size().reset_index()[0]
    female_clear = data[(data['sex'] == 'F') & (data["AF"] == 0)].groupby(['age_range']).size().reset_index()
    age_labels = female_clear["age_range"]
    female_clear = female_clear[0]
    return data, age_ranges, age_labels, female_clear, male_clear, female_sick, male_sick
meta_data, age_ranges, age_labels, fc, mc, fs, ms = split_data_by_age_sex_af(meta_data)
meta_data

maxs = np.max(data, axis=(1, 2))

def set_seed(seed: int = 42) -> None:
    """Set the random seed for reproducibility"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)



def split_id_exam(meta_data, test_split=0.3, min_samples=10, seed=42):
    set_seed(seed)
    # get all the sick males
    age_range_m_sick = meta_data[(meta_data['sex'] == 'M') & (meta_data["AF"] == 1)]
    # get all the heathy males
    age_range_m_clear = meta_data[(meta_data['sex'] == 'M') & (meta_data["AF"] == 0)]

    # get all the sick females
    age_range_f_sick = meta_data[(meta_data['sex'] == 'F') & (meta_data["AF"] == 1)]
    # get all the heathy females
    age_range_f_clear = meta_data[(meta_data['sex'] == 'F') & (meta_data["AF"] == 0)]

    # split data into train and test
    train = []
    test = []

    for age_range_data  in [age_range_m_sick, age_range_m_clear, age_range_f_sick, age_range_f_clear]:
        # groupe each group by age range 
        for age, indexes in age_range_data.groupby(['age_range']).indices.items():
            if len(indexes) == 0:
                continue
            # shuffle the indexes to make a random split
            random.shuffle(indexes)
            # prioritize to get age representation in the test set
            split_index = max(int(len(indexes) * test_split), min_samples)
            test.extend(age_range_data.iloc[indexes[:split_index]].index.values)

            # if there is enough data put the rest in the train set
            if len(indexes) > split_index:
                train.extend(age_range_data.iloc[indexes[split_index:]].index.values)

    print("train: ", len(train))
    print("test: ", len(test))
    assert len(train) + len(test) == len(meta_data)
    train = [ID_TO_INDEX[t] for t in train]
    test = [ID_TO_INDEX[t] for t in test]

    # validate that not train and test have the same id
    assert len(set(train).intersection(set(test))) == 0
    
    return sorted(train), sorted(test) 

def equal_split_id_exam(meta_data, test_split=0.3, min_samples=10, seed=42):
    set_seed(seed)
    # get all the sick males
    age_range_m_sick = meta_data[(meta_data['sex'] == 'M') & (meta_data["AF"] == 1)]
    # get all the heathy males
    age_range_m_clear = meta_data[(meta_data['sex'] == 'M') & (meta_data["AF"] == 0)]

    # get all the sick females
    age_range_f_sick = meta_data[(meta_data['sex'] == 'F') & (meta_data["AF"] == 1)]
    # get all the heathy females
    age_range_f_clear = meta_data[(meta_data['sex'] == 'F') & (meta_data["AF"] == 0)]

    # split data into train and test
    train = []
    test = []

    for age_range_data_sick,  age_range_data_clear in [(age_range_m_sick, age_range_m_clear), (age_range_f_sick, age_range_f_clear)]:
        # groupe each group by age range 
        for (age_sick, indexes_sick), (age_clear, indexes_clear) in zip(age_range_data_sick.groupby(['age_range']).indices.items(), age_range_data_clear.groupby(['age_range']).indices.items()):
            # if there is enough data put the rest in the train set
            if len(indexes_sick) == 0:
                train.extend(age_range_data_clear.iloc[indexes_clear].index.values)
                continue
            # shuffle the indexes to make a random split
            random.shuffle(indexes_sick), random.shuffle(indexes_clear)
            # prioritize to get age representation in the test set
            split_index = max(int(len(indexes_sick) * test_split), min_samples)
            test.extend(age_range_data_sick.iloc[indexes_sick[:split_index]].index.values)
            test.extend(age_range_data_clear.iloc[indexes_clear[:split_index]].index.values)
            train_sick, train_clear = [], []
            if len(indexes_sick) > split_index:
                train_sick = age_range_data_sick.iloc[indexes_sick[split_index:]].index.values
            if len(indexes_clear) > split_index:
                train_clear = age_range_data_clear.iloc[indexes_clear[split_index:]].index.values
                
            if len(train_sick) < len(train_clear) and len(train_sick) > 0:
                len_diff = int(0.5+len(train_clear)/len(train_sick))
                train_sick = np.repeat(train_sick, len_diff)
            elif len(train_sick) > len(train_clear) and len(train_clear) > 0:
                len_diff = int(0.5+len(train_sick)/len(train_clear))
                train_clear = np.repeat(train_clear, len_diff)
            train.extend(train_sick)
            train.extend(train_clear)

    print("train: ", len(train))
    print("test: ", len(test))

    train = [ID_TO_INDEX[t] for t in train]
    test = [ID_TO_INDEX[t] for t in test]

    # validate that not train and test have the same id
    assert len(set(train).intersection(set(test))) == 0, len(set(train).intersection(set(test)))
    
    return sorted(train), sorted(test) 
def validate_data(train_index, val_index, meta_data, dataset, result, func):
    count_lables = 0
    count_array = 0
    for index in train_index + val_index:
        array, label = dataset[index]
        label = int(label)
        if meta_data.iloc[index]['AF'] != label:
            count_lables += 1
            print("wrong lable", index, meta_data.iloc[index]['AF'], label)
        if abs(result[index] - func(array)) > 1e-6:
            count_array += 1
            print("wrong array", index, result[index] - func(array))
    print("wrong lable count: ", count_lables)
    print("wrong array count: ", count_array)

class AugmentIntensity:
    """Normalize the image in a sample.
    """
    def __init__(self, augment_intensity=(0.9, 1.1)):
        self.augment_intensity = augment_intensity

    def __call__(self, sample):
        min_intensity, max_intensity = self.augment_intensity
        tensor, label = sample
        # add random noise to the tensor with values between min_intensity and max_intensity
        tensor = tensor * (torch.rand(tensor.shape) * (max_intensity - min_intensity) + min_intensity)
        return tensor, label
    
class AugmentGaussianNoise:
    """Normalize the image in a sample.
    """
    def __init__(self, augment_noise=(0, 0.01)):
        self.augment_noise = augment_noise

    def __call__(self, sample):
        mean, std = self.augment_noise
        tensor, label = sample
        # add random noise to the tensor 
        tensor = tensor + torch.randn(tensor.shape) * std + mean
        return tensor, label
    
class AugmentShiftData:
    """Normalize the image in a sample.
    """
    def __init__(self, augment_shift=(-0.1, 0.1)):
        self.augment_shift = augment_shift

    def __call__(self, sample):
        min_shift, max_shift = self.augment_shift
        tensor, label = sample
        # shift all values in the tensor with values between min_shift and max_shift
        tensor = torch.roll(tensor, shifts=int(np.random.uniform(min_shift, max_shift) * tensor.shape[0]), dims=0)
        return tensor, label
    
class AugmentScaleData:

    def __init__(self, augment_scale=(0.9, 1.1)):
        self.augment_scale = augment_scale

    def __call__(self, sample):
        min_scale, max_scale = self.augment_scale
        tensor, label = sample
        # scale all values in the tensor with values between min_scale and max_scale
        tensor = tensor * np.random.uniform(min_scale, max_scale)
        return tensor, label

class Identity:
    def __call__(self, sample):
        return sample
    

    
class Aug_Dataset(Dataset):
    def __init__(self, data, transforms=None):
        self.data = data
        self.transforms = transforms

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        array, label = self.data[idx]
        if self.transforms:
            array, label = self.transforms((array, label))
        return array, label
    


augmentations = transforms.RandomChoice([
    AugmentIntensity(), 
    AugmentGaussianNoise(),
    AugmentShiftData(),
    AugmentScaleData(),
    Identity(), 
])

 
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Use device: {device:}\n".format(device=device))

# =============== Build data loaders ======================================#
print("Building data loaders...")

path_to_h5_train, path_to_csv_train, path_to_records = 'codesubset/train.h5', 'codesubset/train.csv', 'codesubset/train/RECORDS.txt'
# load traces
traces = torch.tensor(h5py.File(path_to_h5_train, 'r')['tracings'][()], dtype=torch.float32)
# load labels
ids_traces = [int(x.split('TNMG')[1]) for x in list(pd.read_csv(path_to_records, header=None)[0])] # Get order of ids in traces
labels = torch.tensor(np.array(meta_data['AF'][ids_traces]), dtype=torch.float32).reshape(-1,1)

# create labels for AF (1 or 0), sex (1 or 0), age (0 to 1) from meta_data 
labels_f3 = np.empty((len(meta_data), 3), dtype=np.float32)
labels_f3[:, 0]= np.array(meta_data['AF'])
labels_f3[:, 1] = meta_data['sex'] == 'M'
# normalize age to be between 0 and 1
labels_f3[:, 2] = meta_data['age']  / 100
labels_f3 = torch.tensor(labels_f3, dtype=torch.float32)

# split data
train_index, val_index = equal_split_id_exam(meta_data)
# sanity check that the data is correct and hos not been shuffled since the analysis part
validate_data(train_index, val_index, meta_data, TensorDataset(traces, labels), maxs, lambda x: torch.max(x).item())

dataset_train = Aug_Dataset(TensorDataset(traces[train_index], labels[train_index]), transforms=augmentations)
dataset_valid = TensorDataset(traces[val_index], labels[val_index])

dataset_train_f3 = Aug_Dataset(TensorDataset(traces[train_index], labels_f3[train_index]), transforms=augmentations)
dataset_valid_f3 = TensorDataset(traces[val_index], labels_f3[val_index])

# sanity check that the data amount is correct
assert len(dataset_train) == len(train_index)
assert len(dataset_valid) == len(val_index)
assert len(dataset_train[0][1]) == 1
assert len(dataset_valid[0][1]) == 1

assert len(dataset_train_f3) == len(train_index)
assert len(dataset_valid_f3) == len(val_index)
assert len(dataset_train_f3[0][1]) == 3
assert len(dataset_valid_f3[0][1]) == 3


# build data loaders
train_dataloader= { 
    1: DataLoader(dataset_train,  batch_size=batch_size, shuffle=True), 
    3: DataLoader(dataset_train_f3, batch_size=batch_size, shuffle=True), 
}
valid_dataloader = {
    1: DataLoader(dataset_valid, batch_size=batch_size, shuffle=False),
    3: DataLoader(dataset_valid_f3, batch_size=batch_size, shuffle=False),
}
data_blend = "_standard"
if len(dataset_train) + len(dataset_valid) > len(meta_data):
    data_blend = "_balanced"      
print("Done!\n")

# =============== Optimize ======================================#
db = f"checkpoints/optimize{data_blend}.db"
gpus = 4
possesses = 1
Path(db).parent.mkdir(parents=True, exist_ok=True)
storage_name = f"sqlite:///{db}"
study_name_bl_f1 = "baseline_f1"
study_name_m_f1 = "model_f1"
study_name_m_v2_f1 = "model_v2_f1"

study_name_bl_f3 = "baseline_f3"
study_name_m_f3 = "model_f3"
study_name_m_v2_f3 = "model_v2_f3"


optuna.logging.set_verbosity(optuna.logging.WARNING)
optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize", "maximize"], study_name=study_name_bl_f1, storage=storage_name, load_if_exists=True)
optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize", "maximize"], study_name=study_name_bl_f3, storage=storage_name, load_if_exists=True)

optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize", "maximize"], study_name=study_name_m_f1, storage=storage_name, load_if_exists=True)
optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize", "maximize"], study_name=study_name_m_f3, storage=storage_name, load_if_exists=True)

optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize", "maximize"], study_name=study_name_m_v2_f1, storage=storage_name, load_if_exists=True)
optuna.create_study(directions=["minimize", "minimize", "maximize", "maximize", "maximize"], study_name=study_name_m_v2_f3, storage=storage_name, load_if_exists=True)



for trials, epochs in [(32, 15), (32, 20), (32, 25), (64, 30)]:

    # print(f"epochs: {epochs}, trials: {trials} - baseline f1")
    # optimize(partial(base_model_objective, epochs, train_dataloader, valid_dataloader, 1, data_blend), gpus, possesses, trials, study_name_bl_f1, storage_name)

    # print(f"epochs: {epochs}, trials: {trials} - baseline f3")
    # optimize(partial(base_model_objective, epochs, train_dataloader, valid_dataloader, 3, data_blend), gpus, possesses, trials, study_name_bl_f3, storage_name)

    print(f"epochs: {epochs}, trials: {trials} - model f1")
    optimize(partial(model_objective, epochs, train_dataloader, valid_dataloader, 1, data_blend), gpus, possesses, trials, study_name_m_f1, storage_name)

    print(f"epochs: {epochs}, trials: {trials} - model f3")
    optimize(partial(model_objective, epochs, train_dataloader, valid_dataloader, 3, data_blend), gpus, possesses, trials, study_name_m_f3, storage_name)

    print(f"epochs: {epochs}, trials: {trials} - model_v2 f1")
    optimize(partial(model_v2_objective, epochs, train_dataloader, valid_dataloader, 1, data_blend), gpus, possesses, trials, study_name_m_v2_f1, storage_name)

    print(f"epochs: {epochs}, trials: {trials} - model_v2 f3")
    optimize(partial(model_v2_objective, epochs, train_dataloader, valid_dataloader, 3, data_blend), gpus, possesses, trials, study_name_m_v2_f3, storage_name)


