from __future__ import absolute_import
from typing import List, Dict

import bc.utils.postprocessing as post


def assemble_example() -> str:
    """Returns the assembled string from a simple example"""
    reads: List[str] = ['AAGGCCTAGCT',
                        'AGGCCTAGCAA', 'GGCCTAGCTC', 'AAAGGCCTAGT']
    window_size: int = 100  # arbitrarily chosen
    stride: int = 1  # arbitrarily chosen
    alphabet: Dict[int, str] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    return post.assemble(reads, window_size, stride, alphabet)


def run_all_examples() -> None:
    """Runs and prints all examples in this file."""
    print(assemble_example())
