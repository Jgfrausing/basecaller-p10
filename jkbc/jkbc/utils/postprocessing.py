import difflib
from itertools import groupby
import math
import re
from collections import defaultdict

import numpy as np
from fast_ctc_decode import beam_search
import parasail
import torch

import jkbc.utils.chiron.assembly as chiron
import jkbc.types as t

BLANK_ID = 0
ALPHABET = {0:'-', 1:'A', 2:'C', 3:'G', 4:'T'}
ALPHABET_VALUES = list(ALPHABET.values())
ALPHABET_STR = ''.join(ALPHABET_VALUES)


def calc_accuracy(ref: str, seq: str, balanced=False) -> float:
    """
    Calculate the accuracy between `ref` and `seq`
    """
    alignment = parasail.sw_trace_striped_32(ref, seq, 8, 4, parasail.dnafull)
    counts = defaultdict(int)
    _, cigar = __parasail_to_sam(alignment, seq)

    for count, op  in re.findall(__split_cigar, cigar):
        counts[op] += int(count)

    if balanced:
        accuracy = (counts['='] - counts['I']) / (counts['='] + counts['X'] + counts['D'])
    else:
        accuracy = counts['='] / (counts['='] + counts['I'] + counts['X'] + counts['D'])

    return accuracy * 100


def assemble(reads: t.List[str], window_size: int, stride: int, alphabet: t.Dict[int, str]) -> str:
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

    assembled_with_probabilities: t.List[t.List[float]] = chiron.simple_assembly(
        reads, jump_step_ratio)
    
    assembled_as_numbers: np.ndarray[int] = np.argmax(
        assembled_with_probabilities, axis=0)+1 ## +1 to match A to 1
    
    assembled: str = __concat_str([alphabet[x] for x in assembled_as_numbers])

    return assembled


def convert_idx_to_base_sequence(lst: t.List[int], alphabet: t.List[str] = ALPHABET_VALUES) -> str:
    """Converts a list of base indexes into a str given an alphabet, e.g. [1, 0, 1, 3] -> 'A-AG'"""

    assert max(lst) < len(
        alphabet), "List contains indexes larger than alphabet size - 1."
    assert min(lst) >= 0, "List contains negative indexes."
    
    return __remove_blanks(__concat_str([alphabet[x] for x in lst]))


def decode(predictions: t.Tensor3D, alphabet: str, beam_size: int = 25, threshold: float = 0.1) -> t.List[str]:
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
    assert beam_size > 0 and isinstance(beam_size, int), 'Beam size must be a non-zero positive integer'
    
    # Todo: test what works best
    #predictions = normalise_last_dim(predictions)
    #predictions = normalise_last_dim(torch.nn.LogSoftmax(dim=2)(predictions))
    predictions = torch.nn.Softmax(dim=2)(predictions)
    
    # apply beam search on each window
    decoded: t.List[str] = [beam_search(window.astype(np.float32), alphabet, beam_size, threshold)[0]
                          for window in predictions.cpu().numpy()]
        
    return decoded


# HELPERS

def normalise_last_dim(tensor: t.Tensor3D):
    return (tensor[:,:]-torch.min(tensor[:,:]))/(torch.max(tensor[:,:])-torch.min(tensor[:,:]))


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


def __concat_str(ls: t.List[str]) -> str:
    """Concatenates a list of strings into a single string."""
    return "".join(ls)


def __calc_metrics_from_seq_matcher(seq_matcher: difflib.SequenceMatcher) -> t.Dict[str, int]:
    """Calculate the metrics from a SequenceMatcher and store it in a tag-index dictionary."""
    counts: t.Dict[str, int] = {'insert': 0,
                              'delete': 0, 'equal': 0, 'replace': 0}
    for tag, i1, i2, j1, j2 in seq_matcher.get_opcodes():
        # Look at the inserted range rather than original
        if tag == 'insert':
            counts[tag] += j2 - j1
        else:
            counts[tag] += i2 - i1
    return counts


__split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")

def __parasail_to_sam(result, seq):
    """
    Extract reference start and sam compatible cigar string.

    :param result: parasail alignment result.
    :param seq: query sequence.

    :returns: reference start coordinate, cigar string.
    """    

    cigstr = result.cigar.decode.decode()
    first = re.search(__split_cigar, cigstr)

    first_count, first_op = first.groups()
    prefix = first.group()
    rstart = result.cigar.beg_ref
    cliplen = result.cigar.beg_query

    clip = '' if cliplen == 0 else '{}S'.format(cliplen)
    if first_op == 'I':
        pre = '{}S'.format(int(first_count) + cliplen)
    elif first_op == 'D':
        pre = clip
        rstart = int(first_count)
    else:
        pre = '{}{}'.format(clip, prefix)

    mid = cigstr[len(prefix):]
    end_clip = len(seq) - result.end_query - 1
    suf = '{}S'.format(end_clip) if end_clip > 0 else ''
    new_cigstr = ''.join((pre, mid, suf))
    return rstart, new_cigstr
