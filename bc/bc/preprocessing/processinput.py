from collections import Sized
import h5py as h5py
import numpy as np


class DataCollection(Sized):
    def __init__(self, filename: str, train_validate_split: float = 0.8, min_labels: int = 5, window_size: int = 300, stride: int = 5, min_value = 0, max_value = 850):
        self.filename = filename
        self.train_validate_split = train_validate_split
        self.min_labels = min_labels
        self.pos = 0
        self.test_gen_data = ([], [])
        self.last_train_gen_data = ({}, {})
        self.max_label_len = 50
        self.window_size = window_size
        self.stride = stride
        self.min_value = min_value
        self.max_value = max_value
        with h5py.File(filename, 'r') as h5file:
            self.readIDs = list(h5file['Reads'].keys())

    def get_max_label_len(self) -> int:
        return self.max_label_len

    def __normalise(self, dac, dmin = 0, dmax = 850):
        return (np.clip(dac, dmin, dmax)-dmin)/(dmax-dmin)

    def __standalize(self, dac, mean = 395.27, std = 80, dmin = 0, dmax = 850):
        return list((np.clip(dac, dmin, dmax) - mean) / std)
    
    def __process_read(self, readID: str):
        train_X = []
        train_y = []
        test_X = []
        test_y = []

        with h5py.File(self.filename, 'r') as h5file:
            signal = self.__standalize(h5file['Reads'][readID]['Dacs'][()])
            ref_to_signal = h5file['Reads'][readID]['Ref_to_signal'][()]
            reference = h5file['Reads'][readID]['Reference'][()]

        train_validate_split = round(len(reference)*self.train_validate_split)
        base_count = len(reference)
        start_index = 0
        last_index = base_count-1
        last_start_signal = ref_to_signal[-1]-self.window_size
        for signal_start in range(ref_to_signal[0], last_start_signal, self.stride):
            signal_end = signal_start+self.window_size

            # Get signal of len = window_size
            current_signal = [[x] for x in signal[signal_start:signal_end]]

            # Get labels
            labels = []
            for index in range(start_index, base_count):
                base_location = ref_to_signal[index]
                if base_location < signal_start:
                    # Runtime optimization
                    # Update index such that we don't need to interate over previous bases on next iteration
                    start_index = index+1
                elif base_location > signal_end:
                    # Last base found - Used for train/test-split
                    last_index = index
                    break
                if base_location >= signal_start:
                    # Base is in the window so we add it to labels
                    labels.append(reference[index])

            # Ensure that we have enough labels to train/predict
            if len(labels) > self.min_labels:
                if last_index < train_validate_split:
                    # Adding first part of signal to training
                    train_X.append(current_signal)
                    train_y.append(labels)
                elif start_index > train_validate_split:
                    # ... And last part to test
                    test_X.append(current_signal)
                    test_y.append(labels)
                # else: parts overlapping train_validate_split ignored

        return train_X, train_y, test_X, test_y, reference

    def generator(self):
    """Generator that can be used as iterator.

    Returns:
        training_dict, (test_signal, test_labels)
    """

        for pos in range(len(self)):
            print(f"Processing {self.readIDs[pos]} ({pos})")

            train_x, train_y, test_x, test_y, reference = self.__process_read(
                self.readIDs[pos])

            train_x = np.array(train_x)
            train_y = np.array(train_y)
            test_x = np.array(test_x)
            test_y = np.array(test_y)

            train_X_lens = np.array([[len(x)]
                                     for x in train_x], dtype="float32")
            train_y_lens = np.array([[len(x)]
                                     for x in train_y], dtype="float32")

            # np.array([r + [self.stride]*(self.get_max_label_len()-len(r)) for r in train_y], dtype='float32')
            train_y_padded = train_y
            X = {
                'the_input': train_x,
                'the_labels': train_y,
                'input_length': train_X_lens,
                'label_length': train_y_lens,
                'full_reference' : reference
            }
            #y = {'ctc': np.zeros([len(train_x)])}
            yield X, (test_x, test_y)


    def __len__(self):
        return len(self.readIDs)

    
    
    
    
    
    
    
    
    
    
    
    
    
    