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

    def __init__(self, read_id, x: t.Tensor2D, y: t.Tensor2D, reference: t.Tensor1D):
        self.id = read_id 
        self.x = x
        self.y = y
        self.reference = reference
        
    def x_for_prediction(self, device):
        return torch.tensor(self.x)[:,None].to(device=device)

class SizedTensorDataset(t.TensorDataset):
    r"""Dataset wrapping tensors.

    TensorDataset that returns the tuple (x, (y,y_lengths)).
    CTC_loss must take 3 parameters.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert (len(tensors) == 3)
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        y_lengths = self.tensors[2][index]
        res = (x, (y, y_lengths))

        return res

    def __len__(self):
        return self.tensors[0].size(0)


def convert_to_dataloaders(data: t.Tuple[np.ndarray, np.ndarray, list], split: float, batch_size: int, drop_last: bool = False) -> t.Tuple[t.DataLoader, t.DataLoader]:
    """
    Converts a data object into test/validate TensorDatasets

    Usage:
        train, valid = convert_to_datasets((x, y_padded), split=0.8)
        data = DataBunch.create(train, valid, bs=64)
    """
    # Unpack
    x, y, y_lengths = data
    y_lengths_count = len(y_lengths)
    window_size = x.shape[1]
        
    # Turn it into tensors
    x = torch.as_tensor(x.reshape(x.shape[0], 1, x.shape[1]), dtype = torch.float)
    y = torch.as_tensor(y, dtype = torch.long)
    y_lengths = torch.as_tensor(y_lengths, dtype = torch.long).view(y_lengths_count, 1)
    
    # Get split
    split_train = int(len(x)*split)
    split_valid = split_train-window_size

    # Split into test/valid sets
    x_train_t = x[:split_train]
    y_train_t = y[:split_train]
    y_train_lengths = y_lengths[:split_train]
    x_valid_t = x[split_valid:]
    y_valid_t = y[split_valid:]
    y_valid_lengths = y_lengths[split_valid:,:]

    # Create TensorDataset
    train_ds = SizedTensorDataset(x_train_t, y_train_t, y_train_lengths)
    valid_ds = SizedTensorDataset(x_valid_t, y_valid_t, y_valid_lengths)
    
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
        min_labels_per_window: lower limit for when to discard a window
        window_size: the window size
        stride: how much the moving window moves at a time
    """

    def __init__(self, filename: t.PathLike, max_labels_per_window: int = 70, min_labels_per_window: int = 5,
                 window_size: int = 300, stride: int = 5, training_data=True):

        self.filename = filename
        self.min_labels_per_window = min_labels_per_window
        self.max_labels_per_window = max_labels_per_window
        self.pos = 0
        self.window_size = window_size
        self.stride = stride
        self.training_data = training_data
        with h5py.File(filename, 'r') as h5file:
            self.read_idx = list(h5file['Reads'].keys())

    def __getitem__(self, read_id_index: int) -> ReadObject:
        """
        Returns signal_windows, label_windows, and a reference for a single signal.
        """
        x, y = [], []

        read_id = self.read_idx[read_id_index]

        with h5py.File(self.filename, 'r') as f:
            read = f['Reads'][read_id]
            signal, ref_to_signal, reference = bonito.scale_and_align(read)
        
        num_of_bases = len(reference)
        index_base_start = 0
        last_start_signal = ref_to_signal[-1] - self.window_size
        for window_signal_start in range(ref_to_signal[0], last_start_signal, self.stride):
            # Get a window of the signal
            window_signal_end = window_signal_start + self.window_size
            window_signal = signal[window_signal_start:window_signal_end]

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
                if len(labels) < self.min_labels_per_window: continue
                # And windows exeeding the maximum
                elif len(labels) > self.max_labels_per_window: continue
            x.append(window_signal)
            y.append(labels)
            
        return ReadObject(read_id, x, y, reference)

    def get_range(self, ran: t.List[int], label_len: int, blank_id:int)-> t.Tuple[np.ndarray, np.ndarray, list]:
        x = None
        y = None
        for i in tqdm(ran):
            # Getting data
            data = self[i]
            data_fields = np.array(data.x), np.array(data.y), data.reference
            _x, _y, _ = data_fields # we don't use the full reference while training
            
            # Concating into a single collection
            x = _x if x is None else np.concatenate((x, _x))
            y = _y if y is None else np.concatenate((y, _y))
    
        # Adding padding
        y_lengths = [len(lst) for lst in y]
        y_padded = add_label_padding(labels = y, fixed_label_len = label_len, blank_id=blank_id)

        return (x, y_padded, y_lengths)
    
    def generator(self):
        """Get the next piece of data.
        Returns:
            training_dict, (test_signal, test_labels)
        """

        for pos in range(len(self)):
            yield self[pos]

    def __len__(self):
        return len(self.read_idx)


def add_label_padding(labels: t.Tensor2D, fixed_label_len: int, blank_id: int) -> t.Tensor2D:
    """Pads each label with padding_val

    Example:
        add_label_padding([[1, 1, 2], [3, 4]], fixed_label_len=5, padding_val=0)
        => [[2, 2, 3, 0, 0], [3, 4, 0, 0, 0]]    
        
    Attention:
        will cap label lengths exceding fixed_label_len
    """
    
    return np.array([l + [blank_id] * (fixed_label_len - len(l)) for l in labels], dtype='float32')


def _normalize(dac, dmin: float = 0, dmax: float = 850):
    """Normalize data based on min and max values"""
    return (np.clip(dac, dmin, dmax) - dmin) / (dmax - dmin)


def _standardize(dac, mean: float = 395.27, std: float = 80, dmin: float = 0, dmax: float = 850):
    """Standardize data based on"""
    return list((np.clip(dac, dmin, dmax) - mean) / std)
