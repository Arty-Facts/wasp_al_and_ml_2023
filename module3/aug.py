from torch.utils.data import  Dataset    
import torch
import numpy as np


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