import collections.abc as abc
import functools as ft
import warnings

import h5py as h5py
import numpy as np
import torch


import jkbc.types as t
import jkbc.utils.files as bc_files
import jkbc.utils.postprocessing as pop

BLANK_ID = 0


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

    def __init__(self, x: t.Tensor2D, y: t.Tensor2D, reference: t.Tensor1D):
        self.x = x
        self.y = y
        self.reference = reference


def convert_to_datasets(data: t.Tuple[np.ndarray, np.ndarray], split: float) -> t.Tuple[t.TensorDataset, t.TensorDataset]:
    """
    Converts a data object into test/validate TensorDatasets

    Usage:
        train, valid = convert_to_datasets((x, y_padded), split=0.8)
        data = DataBunch.create(train, valid, bs=64)
    """
    # Unpack
    x, y = data
    window_size = x.shape[1]
    
    # Turn it into tensors
    x_train = torch.tensor(x.reshape(x.shape[0], 1, x.shape[1]), dtype = torch.float)
    y_train = torch.tensor(y, dtype = torch.long)
        
    # Get split
    split_train = int(len(x_train)*split)
    split_valid = split_train-window_size

    # Split into test/valid sets
    x_train_t = x_train[:split_train]
    y_train_t = y_train[:split_train]
    x_valid_t = x_train[split_valid:]
    y_valid_t = y_train[split_valid:]

    # Create TensorDataset
    ds_train = t.TensorDataset(x_train_t, y_train_t)
    ds_valid = t.TensorDataset(x_valid_t, y_valid_t)

    return ds_train, ds_valid


def get_y_lengths(y_pred_len: int, max_y_len: int, batch_size=int) -> t.Tuple[np.ndarray, np.ndarray]:
    prediction_lengths = torch.full(
        size=(batch_size,), fill_value=y_pred_len, dtype=torch.long)
    label_lengths = torch.full(
        size=(batch_size,), fill_value=max_y_len, dtype=torch.long)

    return prediction_lengths, label_lengths


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

    def __init__(self, filename: t.PathLike, min_labels_per_window: int = 5,
                 window_size: int = 300, stride: int = 5):

        self.filename = filename
        self.min_labels_per_window = min_labels_per_window
        self.pos = 0
        self.window_size = window_size
        self.stride = stride
        with h5py.File(filename, 'r') as h5file:
            self.read_idx = list(h5file['Reads'].keys())

    def __getitem__(self, read_id_index: int) -> t.Tuple[t.Tensor2D, t.Tensor2D, t.Tensor1D]:
        """
        Returns signal_windows, label_windows, and a reference for a single signal.
        """
        x, y = [], []

        read_id = self.read_idx[read_id_index]
        print(f"Processing {read_id} ({read_id_index})")

        dac, ref_to_signal, reference = bc_files.get_read_info_from_file(
            self.filename, read_id)
        signal = _standardize(dac)
        
        num_of_bases = len(reference)
        index_base_start = 0
        last_start_signal = ref_to_signal[-1] - self.window_size

        for window_signal_start in range(ref_to_signal[0], last_start_signal, self.stride):
            
            # Get a window of the signal
            window_signal_end = window_signal_start + self.window_size
            window_signal = signal[window_signal_start:window_signal_end]

            # Get labels for current window
            labels = []
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

            x.append(window_signal)
            y.append(labels)
            
        return ReadObject(x, y, reference)

    def get_range(self, ran: range, label_len: int)-> t.Tuple[np.ndarray, np.ndarray, list]:
        x = None
        y = None
        for i in ran:
            # Getting data
            data = self[i]
            data_fields = np.array(data.x), np.array(data.y), data.reference
            _x, _y, _ = data_fields # we don't use the full reference while training

            # Concating into a single collection
            x = _x if x is None else np.concatenate((x, _x))
            y = _y if y is None else np.concatenate((y, _y))
    
        # Adding padding
        y_lengths = [len(lst) for lst in y]
        y_padded = add_label_padding(labels = y, fixed_label_len = label_len)

        return (x, y_padded, y_lengths)
    
    def __iter__(self):
        """Initiates the iterator"""
        self.pos = 0
        return self

    def __next__(self):
        """Receives next piece of data from the file."""
        return self.generator()

    def generator(self):
        """Get the next piece of data.
        Returns:
            training_dict, (test_signal, test_labels)
        """

        for pos in range(len(self)):
            yield self[pos]

    def __len__(self):
        return len(self.read_idx)


def add_label_padding(labels: t.Tensor2D, fixed_label_len: int) -> t.Tensor2D:
    """Pads each label with padding_val until it reaches the fixed_label_length

    Example:
        add_label_padding([[1, 1, 2], [3, 4]], fixed_label_len=5, padding_val=0)
        => [[2, 2, 3, 0, 0], [3, 4, 0, 0, 0]]    
        
    Attention:
        will cap label lengths exceding fixed_label_len
    """
    for length in [len(l) for l in labels if len(l) > fixed_label_len]:
        warnings.warn(f"Capping label length of {length} down to size {fixed_label_len}.")
    capped_labels = [l[:fixed_label_len] for l in labels]
    
    return np.array([l + [BLANK_ID] * (fixed_label_len - len(l)) for l in capped_labels], dtype='float32')


def _normalize(dac, dmin: float = 0, dmax: float = 850):
    """Normalize data based on min and max values"""
    return (np.clip(dac, dmin, dmax) - dmin) / (dmax - dmin)


def _standardize(dac, mean: float = 395.27, std: float = 80, dmin: float = 0, dmax: float = 850):
    """Standardize data based on"""
    return list((np.clip(dac, dmin, dmax) - mean) / std)


def _add_get_state_method(target):
    def get_state(target):
        return{'a':'2'}
    target.method = types.MethodType(get_state,target)
