import unittest
import sys
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

# Add the root directory to the path so we can import the utils module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the module to test
from utils.data_utils import get_dataloaders, get_preprocessed_dataloaders, _WrappedDataLoader


class TestDataUtils(unittest.TestCase):
    """Test cases for data_utils.py module."""

    def setUp(self):
        """Set up common test data."""
        # Create simple tensors for testing
        self.x_train = torch.randn(100, 10)  # 100 samples, 10 features
        self.y_train = torch.randint(0, 2, (100,))  # Binary labels
        
        self.x_valid = torch.randn(20, 10)  # 20 samples, 10 features
        self.y_valid = torch.randint(0, 2, (20,))  # Binary labels
        
        # Create datasets
        self.train_ds = TensorDataset(self.x_train, self.y_train)
        self.valid_ds = TensorDataset(self.x_valid, self.y_valid)

    def test_get_dataloaders_default_params(self):
        """Test get_dataloaders with default parameters."""
        train_dl, valid_dl = get_dataloaders(self.train_ds, self.valid_ds)
        
        # Check types
        self.assertIsInstance(train_dl, DataLoader)
        self.assertIsInstance(valid_dl, DataLoader)
        
        # Check batch sizes
        self.assertEqual(train_dl.batch_size, 64)  # Default batch size
        self.assertEqual(valid_dl.batch_size, 128)  # Default batch size * 2
        
        # Note: DataLoader doesn't expose shuffle as an attribute, so we can't test it directly

    def test_get_dataloaders_custom_params(self):
        """Test get_dataloaders with custom parameters."""
        batch_size = 32
        train_dl, valid_dl = get_dataloaders(
            self.train_ds, self.valid_ds, bs=batch_size, shuffle=False
        )
        
        # Check batch sizes
        self.assertEqual(train_dl.batch_size, batch_size)
        self.assertEqual(valid_dl.batch_size, batch_size * 2)
        
        # Note: DataLoader doesn't expose shuffle as an attribute, so we can't test it directly

    def test_dataloader_iteration(self):
        """Test that the dataloaders can be iterated over."""
        train_dl, valid_dl = get_dataloaders(self.train_ds, self.valid_ds, bs=16)
        
        # Check train dataloader iteration
        batch_count = 0
        for x_batch, y_batch in train_dl:
            batch_count += 1
            self.assertIsInstance(x_batch, torch.Tensor)
            self.assertIsInstance(y_batch, torch.Tensor)
            
            # Check first dimension is less than or equal to batch size
            self.assertLessEqual(x_batch.shape[0], 16)
            self.assertLessEqual(y_batch.shape[0], 16)
            
            # Check other dimensions match our data
            self.assertEqual(x_batch.shape[1], 10)
        
        # We expect ceil(100/16) = 7 batches
        self.assertEqual(batch_count, 7)
        
        # Similar check for validation dataloader
        batch_count = 0
        for x_batch, y_batch in valid_dl:
            batch_count += 1
        
        # We expect ceil(20/32) = 1 batch (batch_size=16*2=32)
        self.assertEqual(batch_count, 1)

    def test_preprocessed_dataloaders(self):
        """Test get_preprocessed_dataloaders."""
        
        # Define a simple preprocessing function
        def preprocess(x, y):
            # Add a channel dimension and normalize x
            x_processed = x.unsqueeze(1) / 255.0
            # One-hot encode y
            y_processed = torch.zeros(y.shape[0], 2)
            y_processed.scatter_(1, y.unsqueeze(1), 1)
            return x_processed, y_processed
        
        # Get preprocessed dataloaders
        train_dl, valid_dl = get_preprocessed_dataloaders(
            self.train_ds, self.valid_ds, preprocess, bs=32
        )
        
        # Check train dataloader
        for x_batch, y_batch in train_dl:
            # Check shapes after preprocessing
            self.assertEqual(x_batch.dim(), 3)  # Added channel dimension
            self.assertEqual(y_batch.shape[1], 2)  # One-hot encoded
            self.assertLessEqual(x_batch.shape[0], 32)  # Batch size
            break  # Only check the first batch
            
        # Check validation dataloader
        for x_batch, y_batch in valid_dl:
            # Check shapes after preprocessing
            self.assertEqual(x_batch.dim(), 3)
            self.assertEqual(y_batch.shape[1], 2)
            self.assertLessEqual(x_batch.shape[0], 64)  # Batch size * 2
            break
    
    def test_wrapped_dataloader_len(self):
        """Test the __len__ method of _WrappedDataLoader."""
        
        def dummy_preprocess(x, y):
            return x, y
            
        train_dl, _ = get_preprocessed_dataloaders(
            self.train_ds, self.valid_ds, dummy_preprocess, bs=20
        )
        
        # For 100 samples with batch_size=20, we expect ceil(100/20) = 5 batches
        self.assertEqual(len(train_dl), 5)
    
    def test_wrapped_dataloader_directly(self):
        """Test the _WrappedDataLoader class directly."""
        # Create a regular DataLoader
        dl = DataLoader(self.train_ds, batch_size=10)
        
        # Define transformation function that modifies both x and y
        def transform_func(x, y):
            return x * 2, y + 1
        
        # Create a wrapped dataloader
        wrapped_dl = _WrappedDataLoader(dl, transform_func)
        
        # Check length
        self.assertEqual(len(wrapped_dl), len(dl))
        
        # Iterate and check transformation
        for original_batch in dl:
            orig_x, orig_y = original_batch
            break
        
        for transformed_x, transformed_y in wrapped_dl:
            # Check that our transformation was applied
            self.assertTrue(torch.allclose(transformed_x, orig_x * 2))
            self.assertTrue(torch.allclose(transformed_y, orig_y + 1))
            break


if __name__ == "__main__":
    unittest.main() 