from collections import deque, Iterable, Sized
import h5py as h5py
import numpy as np
from typing import Union
from processinput import DataCollection

file = 'test_dataset.hdf5'

class DataPreprocessingTest:
        
    def make_test_file(self, reads: int = 5000, bases: int = 100):
        with  h5py.File(file, 'w') as f:
            readId = "readId001"
            f.create_dataset(f'Reads/{readId}/Dacs', data=np.random.rand(reads))
            f.create_dataset(f'Reads/{readId}/Ref_to_signal', data=np.asarray(range(0, reads, reads//bases)))
            f.create_dataset(f'Reads/{readId}/Reference', data=np.asarray([self.__get_test_base(i) for i in range(bases)]))

    def __get_test_base(self, value: int):
        bases = ['A', 'C', 'G','T']
        return value%len(bases)

    def print_dataset(self, filename=file):
        dc = DataCollection(filename)
        (a, b), (c, d) = next(dc)
        print(len(a['the_input']))
        #print(b)
        print(f'c = {c}')
        #print(d)


dpt = DataPreprocessingTest()
dpt.make_test_file()
dpt.print_dataset(filename="/mnt/sdb/taiyaki_mapped/mapped_umi16to9.hdf5")    
dpt.print_dataset()