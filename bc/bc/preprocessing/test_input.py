from typing import Union

import h5py
import numpy as np

from bc.preprocessing import processinput

file = 'test_dataset.hdf5'


class DataPreprocessingTest:

    def make_test_file(self, reads: int = 5000, bases: int = 100):
        with h5py.File(file, 'w') as f:
            readId = "readId001"
            f.create_dataset(f'Reads/{readId}/Dacs',
                             data=np.random.rand(reads))
            f.create_dataset(f'Reads/{readId}/Ref_to_signal',
                             data=np.asarray(range(0, reads, reads//bases)))
            f.create_dataset(f'Reads/{readId}/Reference', data=np.asarray(
                [self.__get_test_base(i) for i in range(bases)]))

    def __get_test_base(self, value: int):
        bases = ['A', 'C', 'G', 'T']
        return value % len(bases)

    def print_dataset(self, filename=file):
        dc = processinput.DataCollection(filename)
        for (a, b), (c, d) in dc.generator():
            print(len(a['the_input']))
            # print(b)
            print(f'c = {c}')
            # print(d)

class SplitFile():
    def __init__(self, filename="/mnt/sdb/taiyaki_mapped/mapped_umi16to9.hdf5"):
        self.pos = 0
        self.filename=filename
        with h5py.File(filename, 'r') as h5file:
            self.readIDs = h5file['Reads'].keys()
        self.size = 10000
        
    def read_file(self):
        with h5py.File(self.filename, 'r') as h5file:
            for readId in self.readIDs:
                signal = h5file['Reads'][readId]['Dacs'][()]
                ref_to_signal = h5file['Reads'][readId]['Ref_to_signal'][()]
                reference = h5file['Reads'][readId]['Reference'][()]
                self.pos+=1
                yield readIlsd, signal, ref_to_signal, reference, self.pos-1
    
    def save(self, filename = "/mnt/sdb/taiyaki_mapped/small_umi16to9", split = 10):
        split = self.size/split
        with h5py.File(f'{filename}', 'w') as f:
            for readId, signal, ref_to_signal, reference, pos in self.read_file():
                if pos>split:
                    return
                f.create_dataset(f'Reads/{readId}/Dacs',
                             data=signal)
                f.create_dataset(f'Reads/{readId}/Ref_to_signal',
                             data=ref_to_signal)
                f.create_dataset(f'Reads/{readId}/Reference', 
                             data=reference)
                                 
def save_range(original="/mnt/sdb/taiyaki_mapped/mapped_umi16to9.hdf5", 
               new = "/mnt/sdb/taiyaki_mapped/small_umi16to9.hdf5", 
               r = 10000):
    with h5py.File(original, 'r') as o:
        with h5py.File(new, 'w') as n:
            g1 = o['Reads']
            read_ids = list(g1.keys())

            for read_id in range(len(read_ids)//10):
                read_id = read_ids[read_id]
                signal = o['Reads'][read_id]['Dacs'][()]
                ref_to_signal = o['Reads'][read_id]['Ref_to_signal'][()]
                reference = o['Reads'][read_id]['Reference'][()]
                
                n.create_dataset(f'Reads/{read_id}/Dacs',
                                 data=signal)
                n.create_dataset(f'Reads/{read_id}/Ref_to_signal',
                                 data=ref_to_signal)
                n.create_dataset(f'Reads/{read_id}/Reference', 
                                 data=reference)
            
            #keys = list(o['Reads'].keys()) #['Reads'][:r]
            
            #print(data)
            #print(o['Reads'].shape)
            #n['Reads'] = data
        
#dpt = DataPreprocessingTest()
#dpt.make_test_file()
#dpt.print_dataset(filename="/mnt/sdb/taiyaki_mapped/mapped_umi16to9.hdf5")
#dpt.print_dataset()

               