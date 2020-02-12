from collections import deque, Iterable, Sized
import h5py
import numpy as np
from typing import Union, Unknown

class DataCollection(Iterable, Sized):
    def __init__(self, filename: str, train_validate_split: float = 0.8, min_labels: int = 5, window_size: int = 300, stride: int = 5):
        self.filename = filename
        self.train_validate_split=train_validate_split
        self.min_labels=min_labels
        self.pos = 0
        self.test_gen_data = ([],[])
        self.last_train_gen_data = ({},{})
        self.max_label_len = 50
        self.window_size = window_size
        self.stride = stride
        with h5py.File(filename, 'r') as h5file:
            self.readIDs = list(h5file['Reads'].keys())

    def get_max_label_len(self):
        return self.max_label_len

    def normalise(self, dac):
        dmin = min(dac)
        dmax = max(dac)
        return [(d-dmin)/(dmax-dmin) for d in dac]

    def __process_read(self, readID: str):
        train_X = []
        train_y = []
        test_X  = []
        test_y  = []
        with h5py.File(self.filename, 'r') as h5file:
            signal = list(self.normalise(h5file['Reads'][readID]['Dacs'][()]))
            ref_to_signal = deque(list(h5file['Reads'][readID]['Ref_to_signal'][()]))
            reference = deque(h5file['Reads'][readID]['Reference'][()])

        train_validate_split = round(len(reference)*(1-self.train_validate_split))
        labels  = deque()
        labelts = deque()

        last_start_signal = ref_to_signal[-1]-self.window_size
        signal_queue = deque()
        for signal_start in range(ref_to_signal[0], last_start_signal, self.stride): 
            signal_end = signal_start+self.window_size
            signal_queue.extend([[x] for x in signal[signal_start:signal_end]])

            while ref_to_signal[0] < signal_end:
                labels.append(reference.popleft())
                labelts.append(ref_to_signal.popleft())

            while len(labelts) > 0 and labelts[0] < signal_end:
                labels.popleft()
                labelts.popleft()

            if len(labels) > self.min_labels:
                if len(ref_to_signal) > train_validate_split:
                    train_X.append(list(signal_queue))
                    train_y.append(list(labels))
                else:
                    test_X.append(list(signal_queue))
                    test_y.append(list(labels))

        return train_X, train_y, test_X, test_y

    def __yield_next_read(self) -> Union[Union[Unknown, Unknown], Union[Unknown, Unknown]]:
        for pos in range(len(self)):
            print(f"Processing {pos}")
            train_x, train_y, test_x, test_y = self.__process_read(self.readIDs[pos])

            train_x = np.array(train_x)
            train_y = np.array(train_y)
            test_x  = np.array(test_x)
            test_y  = np.array(test_y)

            train_X_lens = np.array([[95] for x in train_x], dtype="float32")
            train_y_lens = np.array([[len(x)] for x in train_y], dtype="float32")
            train_y_padded = np.array([r + [5]*(self.get_max_label_len()-len(r)) for r in train_y], dtype='float32')
            X = {
                'the_input': train_x,
                'the_labels': train_y_padded,
                'input_length': train_X_lens,
                'label_length': train_y_lens,
                'unpadded_labels' : train_y
            }
            y = {'ctc': np.zeros([len(train_x)])}
            yield (X, y), (test_x, test_y)

    def test_gen(self):
        while True:
            tgd, self.test_gen_data = self.test_gen_data, ([],[])
            yield tgd


    def __len__(self):
        return len(self.readIDs)

    def __next__(self):
        return next(self.__yield_next_read())
