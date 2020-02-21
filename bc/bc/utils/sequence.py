from typing import List, Dict, Tuple
import difflib

import numpy as np
from fast_ctc_decode import beam_search
import bc.utils.chiron.assembly as chiron
from itertools import groupby


def calc_sequence_error_metrics(actual: str, predicted: str) -> Tuple[float, float, float, float, float]:
    """Calculate several error metrics related to the editdistance between two sequences.

    Args:
        actual: the correct sequence
        predicted: the predicted sequence
    Returns:
        a tuple of metrics in decimal percentage (error, identity, deletion, insertion, mismatch)
    """

    metrics = __calc_metrics_from_seq_matcher(
        difflib.SequenceMatcher(None, actual, predicted))

    len_actual: int = len(actual)
    rate_deletion: float = metrics['delete'] / len_actual
    rate_insertion: float = metrics['insert'] / len_actual
    rate_mismatch: float = metrics['replace'] / len_actual
    rate_identity: float = 1 - rate_deletion - rate_mismatch
    rate_error: float = rate_deletion + rate_insertion + rate_mismatch

    return rate_error, rate_identity, rate_deletion, rate_insertion, rate_mismatch


def assemble(reads: List[str], window_size: int, stride: int, alphabet: Dict[int, str]) -> str:
    """Assemble a list of reads into a string

    Args:
        reads: the reads
        window_size: size of window
        strid: length of stride
        alphabet: dictionary mapping from numbers to letters
    Returns:
        a fully assembled string
    """

    jump_step_ratio: float = stride / window_size

    assembled_with_probabilities: List[List[float]] = chiron.simple_assembly(
        reads, jump_step_ratio)
    assembled_as_numbers: List[int] = np.argmax(
        assembled_with_probabilities, axis=0)
    assembled: str = concat_str([alphabet[x] for x in assembled_as_numbers])
    return assembled

def numeric_to_bases_sequence(lst, alphabet):
    return concat_str([alphabet[x] for x in lst])

def decode(predictions, alphabet, beam_size=5, threshold=0.1):
    """
    Decode model posteriors to sequence
    """
    alphabet = concat_str(alphabet)
    
    decoded = [beam_search(window.astype(np.float32), alphabet, beam_size, threshold) for window in predictions]
    decoded = [remove_duplicates(seq) for seq in decoded]
    
    return [seq.replace('-','') for seq in decoded]

# HELPERS
def remove_duplicates(s):
     return (''.join(i for i, _ in groupby(s)))

def split(word): 
    return [char for char in word]  
      
def concat_str(ls: List[str]) -> str:
    """Concatenates a list of strings into a single string."""
    return "".join(ls)


def __calc_metrics_from_seq_matcher(seq_matcher: difflib.SequenceMatcher) -> Dict[str, int]:
    """Calculate the metrics from a SequenceMatcher and store it in a tag-index dictionary. (Warning: Inefficient)"""
    counts: Dict[str, int] = {'insert': 0,
                              'delete': 0, 'equal': 0, 'replace': 0}
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        # Look at the inserted range rather than original
        if tag == 'insert':
            counts[tag] += j2 - j1
        else:
            counts[tag] += i2 - i1
    return counts


