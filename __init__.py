"""
ML utilities for data processing and model training.
"""

# Import and re-export frequently used ML modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

import torchvision
from torchvision import datasets, transforms


from torch.utils.tensorboard.writer import SummaryWriter
from torchinfo import summary


# Make these modules available when importing from utils
__all__ = ['torch', 'nn', 'F', 'optim', 'DataLoader', 'Dataset', 'TensorDataset', 'summary', 'SummaryWriter', 'datasets', 'transforms', 'torchvision']


# Import and re-export the functions from data_utils
from .data_utils import get_dataloaders, get_preprocessed_dataloaders

# Add any other functions you want to expose directly
# from .other_module import other_function

__all__ += ['get_dataloaders', 'get_preprocessed_dataloaders'] 

from .visualize_utils import get_tensorboard_writer, add_images_to_tensorboard, matplotlib_imshow, log_model_graph, log_scalars, log_scalar, log_embedding, log_histograms, close_writer

__all__ += ['get_tensorboard_writer', 'add_images_to_tensorboard', 'matplotlib_imshow', 'log_model_graph', 'log_scalars', 'log_scalar', 'log_embedding', 'log_histograms', 'close_writer']

from .torch_utils import use_cuda
__all__ += ['use_cuda']
