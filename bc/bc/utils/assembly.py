from typing import List, Dict

import numpy as np

import bc.utils.chiron.assembly as chiron


def calculate_error(actual: str, predicted: str) -> float:
    """Not yet implemented"""
    return 0


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


def concat_str(ls: List[str]) -> str:
    """Concatenates a list of strings into a single string."""
    return "".join(ls)
