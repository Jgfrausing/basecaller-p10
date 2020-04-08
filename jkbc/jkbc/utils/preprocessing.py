import collections.abc as abc
import functools as ft
import warnings

import h5py as h5py
import numpy as np
import torch
from tqdm import tqdm


import jkbc.types as t
import jkbc.utils.files as bc_files
import jkbc.utils.constants as constants
import jkbc.utils.bonito.data as bonito


class ReadObject:
    """
    Contains the relevant information from reads for a single signal.

    Args:
        x: [Window][SignalValue]
        y: [Window][Base]
        x_lengths: the number of signals in each window
        y_lengths: the number of bases in each window
        reference: the full genome reference
    """

    def __init__(self, read_id, x: t.Tensor2D, x_lengths: t.List[int], y: t.Tensor2D, y_lengths: t.List[int], reference: t.Tensor1D):
        assert len(x) == len(x_lengths), "Dimensions of input parameters does not fit"
        assert len(y) == len(y_lengths), "Dimensions of output parameters does not fit"
        assert len(y) == 0 or len(y) == len(x), 'y contains elements (e.g. training data is included) but is not same size as x'
        self.id = read_id 
        self.x = x
        self.x_lengths = x_lengths
        self.y = y
        self.y_lengths = y_lengths
        self.reference = reference

class SizedTensorDataset(t.TensorDataset):
    r"""Dataset wrapping tensors.

    TensorDataset that returns the tuple (x, (y,y_lengths)).
    CTC_loss must take 3 parameters.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """
    def __init__(self, *tensors):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors), "All tensors does not have same size"
        self.tensors = tensors
        self.has_teacher = len(tensors) == 5
                
    def __getitem__(self, index):
        x = self.tensors[0][index]
        x_lengths = self.tensors[1][index]
        y = self.tensors[2][index]
        y_lengths = self.tensors[3][index]
        
        if self.has_teacher:
            second = (y, x_lengths, y_lengths, self.tensors[4][index])
        else:
            second = (y, x_lengths, y_lengths) 
        
        return (x, second)

    def __len__(self):
        return self.tensors[0].size(0)


def convert_to_dataloaders(data: ReadObject, split: float, batch_size: int, teacher:t.Tensor=None, drop_last: bool = False, windows:int=None) -> t.Tuple[t.DataLoader, t.DataLoader]:
    """
    Converts a data object into test/validate TensorDatasets

    Usage:
        train, valid = convert_to_datasets((x, y_padded), split=0.8)
        data = DataBunch.create(train, valid, bs=64)
    """
    if not windows:
        windows = len(data.x)
    
    x = data.x[:windows]
    x_lengths = data.x_lengths[:windows]
    y = data.y[:windows]
    y_lengths = data.y_lengths[:windows]
    
    window_size = len(x[0])
        
    # Turn it into tensors
    x = torch.as_tensor(x, dtype = torch.float).view(windows, 1, window_size)
    x_lengths = torch.as_tensor(x_lengths, dtype = torch.long).view(windows, 1)
    
    y = torch.as_tensor(y, dtype = torch.long)
    y_lengths = torch.as_tensor(y_lengths, dtype = torch.long).view(windows, 1)
    
    # Get split
    split = int(windows*split)

    # Split into test/valid sets
    x_train_t = x[:split]
    x_train_lengths = x_lengths[:split]
    y_train_t = y[:split]
    y_train_lengths = y_lengths[:split]
    
    x_valid_t = x[split:]
    x_valid_lengths = x_lengths[split:,:]
    y_valid_t = y[split:]
    y_valid_lengths = y_lengths[split:,:]

    # Create TensorDataset
    if not teacher:
        train_ds = SizedTensorDataset(x_train_t, x_train_lengths, y_train_t, y_train_lengths)
        valid_ds = SizedTensorDataset(x_valid_t, x_valid_lengths, y_valid_t, y_valid_lengths)
    else: ## ERROR STARTS SOMEWHERE HERE!
        y_train_teacher = y_teacher[:split_train]
        y_valid_teacher = y_teacher[split_valid:]
        
        train_ds = SizedTensorDataset(x_train_t, x_train_lengths, y_train_t, y_train_lengths, y_train_teacher)
        valid_ds = SizedTensorDataset(x_valid_t, x_valid_lengths, y_valid_t, y_valid_lengths, y_valid_teacher)
        
    # Create DataLoader
    train_dl = t.DataLoader(train_ds, batch_size=batch_size, drop_last=drop_last)
    valid_dl = t.DataLoader(valid_ds, batch_size=batch_size, drop_last=drop_last)

    return train_dl, valid_dl


def get_prediction_lengths(y_pred_len: int, batch_size: int) -> t.Tuple[np.ndarray, np.ndarray]:
    prediction_lengths = torch.full(
        size=(batch_size,), fill_value=y_pred_len, dtype=torch.long)
    
    return prediction_lengths


# TODO: Implement missing Generator functions
class SignalCollection(abc.Sequence):
    """
    An iterator and generator for getting signal data from HDF5 files.

    Args:
        filename: The path to the HDF5 file
        labels_per_window: tuple of lower and upper limit for when to discard a window
        window_size: tuple of lower and upper limit of the window size
        stride: how much the moving window moves at a time
        training_data: to include y values or not
    """

    def __init__(self, filename: t.PathLike, window_size: t.Tuple[int, int], blank_id: int, 
                 stride: int = 5, labels_per_window: t.Tuple[int, int] = None, training_data=True):

        assert not training_data or labels_per_window != None, "labels_per_window must be set to create training data"
        
        self.filename = filename
        self.labels_per_window = labels_per_window
        self.window_size = window_size
        self.blank_id = blank_id
        self.stride = stride
        self.training_data = training_data
        
        with h5py.File(filename, 'r') as h5file:
            self.read_idx = list(h5file['Reads'].keys())

    def __getitem__(self, read_id_index: int) -> ReadObject:
        """
        Returns signal_windows, label_windows, and a reference for a single signal.
        """
        x, x_lengths = [], []
        y, y_lengths = [], []

        read_id = self.read_idx[read_id_index]

        with h5py.File(self.filename, 'r') as f:
            read = f['Reads'][read_id]
            signal, ref_to_signal, reference = bonito.scale_and_align(read)
        
        num_of_bases = len(reference)
        index_base_start = 0
        
        for window_signal_start in range(ref_to_signal[0], ref_to_signal[-1], self.stride):
            # Get a window of the signal
            window_signal_end = window_signal_start + np.random.randint(self.window_size[0], self.window_size[1])
            window_signal_end = min(window_signal_end, ref_to_signal[-1])
            
            window_signal = signal[window_signal_start:window_signal_end]
            # Continue if window is too small
            if len(window_signal) < self.window_size[0]:
                continue
            if len(window_signal) > self.window_size[1]:
                continue
                        
            # Get labels for current window
            labels = []
            if self.training_data:
                for index_base in range(index_base_start, num_of_bases):
                    base_location = ref_to_signal[index_base]
                    if base_location < window_signal_start:
                        # Update index_base such that we don't need to interate over previous bases on next iteration
                        index_base_start = index_base + 1
                    elif base_location >= window_signal_end:
                        # base_location is beyond the current window
                        break
                    else:
                        # Base is in the window so we add it to labels
                        # One is added to avoid As and BLANKs (index 0) clashing in CTC
                        labels.append(reference[index_base]+1)

                # Discard windows with very few corresponding labels
                if len(labels) < self.labels_per_window[0]: continue
                # And windows exeeding the maximum
                elif len(labels) > self.labels_per_window[1]: continue
            
            # Get lengths for signals and labels before padding
            x_lengths.append(len(window_signal))
            y_lengths.append(len(labels))
            
            # Padding input with zeros
            window_signal = add_padding(window_signal, self.window_size[1], 0)
            
            # Padding labels with blank_id
            if self.training_data:
                labels = add_padding(labels, self.labels_per_window[1], self.blank_id)
            
            # Append to signals and labels
            x.append(window_signal)
            y.append(labels)
        
        return ReadObject(read_id, x, x_lengths, y, y_lengths, reference)

    
    def get_range(self, ran: t.List[int])-> t.Tuple[np.ndarray, np.ndarray, list]:
        x, x_lengths = None, None
        y, y_lengths = None, None
        
        for i in tqdm(ran):
            # Getting data
            data = self[i]
            _x, _x_lengths, _y, _y_lengths = data.x, data.x_lengths, data.y, data.y_lengths
            
            # Concating into a single collection
            if x is None:
                x, x_lengths = _x, _x_lengths
                y, y_lengths = _y, _y_lengths
            else:
                x += _x
                x_lengths += _x_lengths
                y += _y
                y_lengths += _y_lengths
                        
        assert len(x) == len(y) == len(x_lengths) == len(y_lengths), "Dimensions does not match for training data"
        return ReadObject(None, x, x_lengths, y, y_lengths, None)
    
    def generator(self) -> ReadObject:
        """Get the next piece of data.
        Returns:
            ReadObject
        """

        for pos in range(len(self)):
            yield self[pos]

    def __len__(self):
        return len(self.read_idx)


def add_padding(lst: t.List[int], length: int, padding_id: int) -> t.List[int]:
    assert len(lst) <= length, f"Cannot pad lst longer than given length {len(lst), length}"
    return np.append(lst, [padding_id] * (length - len(lst)))


def _normalize(dac, dmin: float = 0, dmax: float = 850):
    """Normalize data based on min and max values"""
    return (np.clip(dac, dmin, dmax) - dmin) / (dmax - dmin)


def _standardize(dac, mean: float = 395.27, std: float = 80, dmin: float = 0, dmax: float = 850):
    """Standardize data based on"""
    return list((np.clip(dac, dmin, dmax) - mean) / std)
