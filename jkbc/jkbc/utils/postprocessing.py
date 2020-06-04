from itertools import groupby

from fast_ctc_decode import beam_search
import numpy as np
import torch

import jkbc.types as t
import jkbc.utils.chiron.assembly as chiron
import jkbc.utils.bonito.decode as bonito


class PredictObject():
    def __init__(self, id, bacteria, predictions, references, assembled, full_reference):
        self.id = id
        self.bacteria = bacteria
        self.predictions = predictions
        self.references = references
        self.assembled = assembled
        self.full_reference = full_reference


def calc_accuracy(ref: str, seq: str, return_alignment=False) -> t.Union[float, t.Tuple[float, str]]:
    return bonito.accuracy(ref, seq, return_alignment)


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


def convert_idx_to_base_sequence(lst: t.List[int], alphabet: t.List[str]) -> str:
    """Converts a list of base indexes into a str given an alphabet, e.g. [1, 0, 1, 3] -> 'A-AG'"""

    if len(lst) == 0:
        return []
    assert max(lst) < len(
        alphabet), "List contains indexes larger than alphabet size - 1."
    assert min(lst) >= 0, "List contains negative indexes."
    
    return __remove_blanks(__concat_str([alphabet[int(x)] for x in lst]))


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
    
    predictions = torch.nn.Softmax(dim=2)(predictions.to(dtype=torch.float32)).cpu().numpy()
    
    # apply beam search on each window
    decoded: t.List[str] = [beam_search(window.astype(np.float32), alphabet, beam_size, threshold)[0] for window in predictions]
        
    return decoded


# HELPERS

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
