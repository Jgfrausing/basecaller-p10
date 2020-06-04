import numpy as np
import h5py

def __align(samples, pointers, reference):
    """ align to the start of the mapping """
    squiggle_duration = len(samples)
    mapped_off_the_start = len(pointers[pointers < 0])
    mapped_off_the_end = len(pointers[pointers >= squiggle_duration])
    pointers = pointers[mapped_off_the_start:len(pointers) - mapped_off_the_end]
    reference = reference[mapped_off_the_start:len(reference) - mapped_off_the_end]
    return samples[pointers[0]:pointers[-1]], pointers - pointers[0], reference

def __scale(read, samples, normalise=True):
    """ scale and normalise a read """
    scaling = read.attrs['range'] / read.attrs['digitisation']
    scaled = (scaling * (samples + read.attrs['offset'])).astype(np.float32)
    if normalise:
        return (scaled - read.attrs['shift_frompA']) / read.attrs['scale_frompA']
    return scaled

# TODO: Use if AL decides, that it is usefull to collapse homopolymers
def __boundary(sequence, r=5):
    """ check if we are on a homopolymer boundary """
    return len(set(sequence[-r:])) == 1

def scale_and_align(read):
    reference = read['Reference'][:]
    pointers = read['Ref_to_signal'][:]
    samples = read['Dacs'][:]
    samples = __scale(read, samples)
    samples, pointers, reference = __align(samples, pointers, reference)
    
    return samples, pointers, reference