"""
Utilities for data loading and preprocessing in PyTorch.

This module provides helper functions to create and manage DataLoaders
with optional preprocessing capabilities.
"""
from typing import Tuple, Callable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


__all__ = ['get_dataloaders', 'get_preprocessed_dataloaders']


def get_dataloaders(
    train_ds: TensorDataset,
    valid_ds: TensorDataset,
    bs: int = 64,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader objects for training and validation datasets.
    
    Args:
        train_ds: Training dataset
        valid_ds: Validation dataset
        bs: Batch size for training (validation uses 2x this value)
        shuffle: Whether to shuffle the training data
        
    Returns:
        Tuple of (training DataLoader, validation DataLoader)
    """
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=shuffle),
        DataLoader(valid_ds, batch_size=bs * 2),
    )


def get_preprocessed_dataloaders(
    train_ds: TensorDataset,
    valid_ds: TensorDataset,
    preprocess: Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]],
    bs: int = 64,
    shuffle: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create DataLoader objects with preprocessing applied to each batch.
    
    Args:
        train_ds: Training dataset
        valid_ds: Validation dataset
        preprocess: Function to apply to each batch (x, y) -> (new_x, new_y)
        bs: Batch size for training (validation uses 2x this value)
        shuffle: Whether to shuffle the training data
        
    Returns:
        Tuple of (training DataLoader, validation DataLoader) with preprocessing
    
    Example:
        ```python
        # Example preprocessing function
        def preprocess(x, y):
            return x.view(-1, 1, 28, 28), y
        
        train_dl, valid_dl = get_preprocessed_dataloaders(train_ds, valid_ds, preprocess)
        ```
    """
    train_dl, valid_dl = get_dataloaders(train_ds, valid_ds, bs, shuffle)
    train_dl = _WrappedDataLoader(train_dl, preprocess)
    valid_dl = _WrappedDataLoader(valid_dl, preprocess)
    return train_dl, valid_dl


class _WrappedDataLoader:
    """
    A wrapper around DataLoader to apply a preprocessing function to each batch.
    """
    
    def __init__(self, dl: DataLoader, func: Callable):
        """
        Initialize the wrapped DataLoader.
        
        Args:
            dl: DataLoader to wrap
            func: Function to apply to each batch
        """
        self.dl = dl
        self.func = func

    def __len__(self) -> int:
        """Return the number of batches in the DataLoader."""
        return len(self.dl)

    def __iter__(self):
        """Iterate through batches, applying the preprocessing function to each."""
        for batch in self.dl:
            yield self.func(*batch)