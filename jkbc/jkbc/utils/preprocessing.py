import collections

import h5py as h5py
import numpy as np

import jkbc.types as t
import jkbc.utils.files as bc_files

BLANK_ID = 4


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

    def __init__(self, x: t.Tensor2D, y: t.Tensor2D, x_lengths: t.Tensor1D, y_lengths: t.Tensor1D,
                 reference: t.Tensor1D):
        self.x = x
        self.y = y
        self.x_lengths = x_lengths
        self.y_lengths = y_lengths
        self.reference = reference


# TODO: Implement missing Generator functions
class SignalCollection(collections.Sequence, collections.Generator):
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

        signal, ref_to_signal, reference = bc_files.get_read_info_from_file(self.filename, read_id)

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
                    labels.append(reference[index_base])

            # Discard windows with very few corresponding labels
            if len(labels) < self.min_labels_per_window: break

            x.append(window_signal)
            y.append(labels)

        return x, y, reference

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
            print(f"Processing {self.read_idx[pos]} ({pos})")

            x, y, reference = self[pos]

            x_lengths = np.array([[len(x_)] for x_ in x], dtype="float32")
            y_lengths = np.array([[len(y_)] for y_ in y], dtype="float32")

            # Maybe this is needed? Felix used it.
            # y_pred = {'ctc': np.zeros([len(train_x)])}

            yield ReadObject(x, y, x_lengths, y_lengths, reference)

    def __len__(self):
        return len(self.read_idx)


def add_label_padding(labels: t.Tensor2D, fixed_label_len: int, padding_id: int = BLANK_ID) -> t.Tensor2D:
    """Pads each label with padding_id until it reaches the fixed_label_length

    Example:
        add_label_padding([[1, 2, 3], [2, 3]], fixed_label_len=5, padding_id=4)
        => [[1, 2, 3, 4, 4], [2, 3, 4, 4, 4]]    
    """
    return np.array([l + [padding_id] * (fixed_label_len - len(l)) for l in labels], dtype='float32')

# TODO: Complete type signature
def __normalize(dac, dmin: float = 0, dmax: float = 850):
    """Normalize data based on min and max values"""
    return (np.clip(dac, dmin, dmax) - dmin) / (dmax - dmin)

# TODO: Complete type signature
def __standardize(dac, mean: float = 395.27, std:float = 80, dmin:float = 0, dmax: float = 850):
    """Standardize data based on"""
    return list((np.clip(dac, dmin, dmax) - mean) / std)
