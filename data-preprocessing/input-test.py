from collections import deque, Iterable, Sized
import h5py as h5py
import numpy as np
from typing import Union
from processinput import DataCollection

file = 'test_dataset.hdf5'

class DataPreprocessingTest:
    
    def make_test_file(self, reads: int = 500, bases: int = 100):
        with  h5py.File(file, 'w') as f:

            f.create_dataset('Reads/Dacs', data=np.random.rand(reads))
            f.create_dataset('Reads/Ref_to_signal', data=np.asarray(range(0, reads, reads//bases)))
            reference = ''.join([self.__get_test_base(i) for i in range(bases)])
            f.create_dataset('Reads/Reference', data=np.string_(reference))

    def __create_dataset(self, reads: int = 500, bases: int = 100):
        data = dict()
        data["readId_1"]['Dacs'] = np.random.rand(reads)
        data["readId_1"]['Reference'] = [self.__get_test_base(i) for i in range(bases)]
        data["readId_1"]['Ref_to_signal'] = range(0, reads, reads//bases)

    def __get_test_base(self, value: int):
        bases = ['A', 'C', 'G','T']
        return bases[value%len(bases)]

    def print_dataset(self, filename=file):
        dc = DataCollection(file)
        #nxt = next(dc)
        print(dc)


dpt = DataPreprocessingTest()
#dpt.make_test_file()
dpt.print_dataset("/mnt/sdb/taiyaki_mapped/mapped_umi16to9.hdf5")    
