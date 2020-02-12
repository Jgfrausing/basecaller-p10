from unittest import TestCase
from typing import List, Dict

import bc.utils.assembly as assembly


class TestAssembly(TestCase):
    def test_assembles_correctly(self):
        # Arrange
        reads: List[str] = ['AAGGCCTAGCT',
                            'AGGCCTAGCAA', 'GGCCTAGCTC', 'AAAGGCCTAGT']
        window_size: int = 100  # arbitrarily chosen
        stride: int = 1  # arbitrarily chosen
        alphabet: Dict[int, str] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

        # Act
        result: str = assembly.assemble(reads, window_size, stride, alphabet)

        # Assert
        self.assertEqual("AAGGCCTAGCTA", result)
