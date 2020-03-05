from typing import NewType, Union, Tuple
import pathlib as pl

import numpy as np
from torch.utils.data import TensorDataset

PathLike = NewType("PathLike", Union[pl.Path, str])

Tensor1D = NewType("Tensor1D", np.ndarray)
Tensor2D = NewType("Tensor2D", np.ndarray)
Tensor3D = NewType("Tensor3D", np.ndarray)