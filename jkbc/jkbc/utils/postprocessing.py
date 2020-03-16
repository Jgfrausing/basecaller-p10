import difflib
from itertools import groupby
import math
from typing import List, Dict

import numpy as np
from fast_ctc_decode import beam_search

import jkbc.utils.chiron.assembly as chiron
import jkbc.types as t

ALPHABET = {0:'-', 1:'A', 2:'C', 3:'G', 4:'T'}
ALPHABET_VALUES = list(ALPHABET.values())
ALPHABET_STR = ''.join(ALPHABET_VALUES)


class Rates:
    """A 'struct' containing the different types of rates (deletion, insertion, mismatch, identity, error)"""

    def __init__(self, deletion: float, insertion: float, mismatch: float, identity: float, error: float):
        self.deletion = deletion
        self.insertion = insertion
        self.mismatch = mismatch
        self.identity = identity
        self.error = error


def calc_sequence_error_metrics(actual: str, predicted: str) -> Rates:
    """Calculate several error metrics related to the edit-distance between two sequences.

    Args:
        actual: the correct sequence
        predicted: the predicted sequence
    Returns:
        a Rates object with each of the different types of rates as fields
    """

    metrics = __calc_metrics_from_seq_matcher(
        difflib.SequenceMatcher(None, actual, predicted))

    len_actual: int = len(actual)
    rate_deletion: float = metrics['delete'] / len_actual
    rate_insertion: float = metrics['insert'] / len_actual
    rate_mismatch: float = metrics['replace'] / len_actual
    rate_identity: float = 1 - rate_deletion - rate_mismatch
    rate_error: float = rate_deletion + rate_insertion + rate_mismatch

    return Rates(rate_deletion, rate_insertion, rate_mismatch, rate_identity, rate_error)


def assemble(reads: List[str], window_size: int, stride: int, alphabet: Dict[int, str] = ALPHABET) -> str:
    """Assemble a list of reads into a string

    Args:
        reads: the reads
        window_size: size of window
        stride: length of stride
        alphabet: dictionary mapping from numbers to letters
    Returns:
        a fully assembled string
    """
    jump_step_ratio: float = stride / window_size

    assembled_with_probabilities: List[List[float]] = chiron.simple_assembly(
        reads, jump_step_ratio)
    assembled_as_numbers: np.ndarray[int] = np.argmax(
        assembled_with_probabilities, axis=0)
    assembled: str = __concat_str([alphabet[x] for x in assembled_as_numbers])

    return assembled


def convert_idx_to_base_sequence(lst: List[int], alphabet: str = ALPHABET_VALUES) -> str:
    """Converts a list of base indexes into a str given an alphabet, e.g. [1, 0, 1, 3] -> 'A-AG'"""

    assert max(lst) < len(
        alphabet), "List contains indexes larger than alphabet size - 1."
    assert min(lst) >= 0, "List contains negative indexes."

    return __concat_str([alphabet[x] for x in lst])


def decode(predictions: t.Tensor3D, alphabet: str = ALPHABET_STR, beam_size: int = 25, threshold: float = 0.1, predictions_in_log: bool = True) -> List[str]:
    """Decode model posteriors to sequence.

    Args:
        predictions: the reads are [WindowCount][WindowSize][AlphabetSize]
        alphabet: str of ordered labels
        beam_size: the number of candidates to consider
        threshold: characters below this threshold are not considered
        predictions_in_log: is output from network in log?
    Returns:
        a decoded string
    """

    if predictions_in_log:
        predictions = convert_logsoftmax_to_softmax(predictions)
    # apply beam search on each window
    decoded: List[str] = [beam_search(window.astype(np.float32), alphabet, beam_size, threshold)[0]
                          for window in predictions]
        
    return decoded


# HELPERS

def convert_logsoftmax_to_softmax(log_softmax_tensor: np.ndarray) -> np.ndarray:
    """Use before beam search"""
    return pow(math.e,log_softmax_tensor)


def __remove_duplicates_and_blanks(s: str, blank_char: str = '-') -> str:
    """Removes duplicates first and then blanks, e.g. 'AA--ATT' -> 'A-AT' -> 'AAT'"""
    return __remove_blanks(__remove_duplicates(s), blank_char=blank_char)


def __remove_blanks(s: str, blank_char: str = '-') -> str:
    """Removes blanks from a string, e.g. 'A-GA-T' -> 'AGAT'."""
    return s.replace(blank_char, '')


def __remove_duplicates(s: str) -> str:
    """Removes duplicates in a string, e.g. 'AA--ATT' -> 'A-AT'."""
    return ''.join(i for i, _ in groupby(s))


def __concat_str(ls: List[str]) -> str:
    """Concatenates a list of strings into a single string."""
    return "".join(ls)


def __calc_metrics_from_seq_matcher(seq_matcher: difflib.SequenceMatcher) -> Dict[str, int]:
    """Calculate the metrics from a SequenceMatcher and store it in a tag-index dictionary."""
    counts: Dict[str, int] = {'insert': 0,
                              'delete': 0, 'equal': 0, 'replace': 0}
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        # Look at the inserted range rather than original
        if tag == 'insert':
            counts[tag] += j2 - j1
        else:
            counts[tag] += i2 - i1
    return counts
