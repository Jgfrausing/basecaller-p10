from unittest import TestCase
from typing import List, Dict

import jkbc.utils.postprocessing as postprocessing


class TestPostprocessing(TestCase):
    def test_assembles_correctly(self):
        # Arrange
        reads: List[str] = ['AAGGCCTAGCT',
                            'AGGCCTAGCAA', 'GGCCTAGCTC', 'AAAGGCCTAGT']
        window_size: int = 100  # arbitrarily chosen
        stride: int = 1  # arbitrarily chosen
        alphabet: Dict[int, str] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

        # Act
        result: str = postprocessing.assemble(reads, window_size, stride, alphabet)

        # Assert
        self.assertEqual("AAGGCCTAGCTA", result)

    def test_calcs_correct_identity(self):
        # Arrange
        s1: str = 'AAGGCC'
        s2: str = 'AAAGGCCA'

        # Act
        metrics = postprocessing.__calc_sequence_error_metrics(s1, s2)
        _, rate_identity, _, _, _ = metrics

        # Assert
        self.assertEqual(rate_identity, 1)
