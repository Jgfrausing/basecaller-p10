from typing import *
import pathlib as pl

import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor

PathLike = NewType("PathLike", Union[pl.Path, str])

Tensor1D = NewType("Tensor1D", np.ndarray)
Tensor2D = NewType("Tensor2D", np.ndarray)
Tensor3D = NewType("Tensor3D", np.ndarray)

DataCollection = NewType("DataCollection", Union[Tuple[np.ndarray, np.ndarray, list], Tuple[np.ndarray, np.ndarray, list, np.ndarray]])
