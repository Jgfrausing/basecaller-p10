import os
import pathlib as pl
import feather as feather
import pandas as pd

import h5py
import numpy as np

import jkbc.types as t


def make_dummy_hdf5_file(file_path: t.PathLike = pl.Path("test_dataset.hdf5"), reads: int = 5000,
                         base_count: int = 100) -> None:
    """Creates a hdf5 file with dummy values for signal, ref2sig, and references."""
    fake_signal = np.random.rand(reads)
    fake_ref_to_signal = np.asarray(range(0, reads+1, reads // base_count))
    fake_reference = np.zeros(base_count)

    with h5py.File(file_path, 'w') as file_new:
        read_id = "readId001"
        write_read_info_to_open_file(
            file_new, read_id, fake_signal, fake_ref_to_signal, fake_reference)


def copy_part_of_file_to(filepath_original: t.PathLike = "/mnt/sdb/taiyaki_mapped/mapped_umi16to9.hdf5",
                         filepath_new: t.PathLike = "/mnt/sdb/taiyaki_mapped/small_umi16to9.hdf5",
                         dec_percentage: float = 0.1):
    """Copies some percentage (0. - 1.) of a HDF5 file into a new file."""

    assert 0.0 <= dec_percentage <= 1.0, "dec_percentage must be given in the range 0.0 to 1.0"

    with h5py.File(filepath_original, 'r') as file_original:
        with h5py.File(filepath_new, 'w') as file_new:
            g1 = file_original['Reads']
            read_ids = list(g1.keys())

            for read_id in range(round(len(read_ids) * dec_percentage)):
                read_id = read_ids[read_id]
                signal, ref_to_signal, reference = get_read_info_from_open_file(
                    file_original, read_id)
                write_read_info_to_open_file(
                    file_new, read_id, signal, ref_to_signal, reference)


def write_read_info_to_open_file(file: h5py.File, read_id: str, signal: np.ndarray, ref_to_signal: np.ndarray,
                                 reference: np.ndarray) -> None:
    """Writes the read-info to an open HDF5 file."""
    assert len(ref_to_signal)-1 == len(
        reference), "ref_to_signal must contain exactly one more element than reference"

    file.create_dataset(f'Reads/{read_id}/Dacs',
                        data=signal)
    file.create_dataset(f'Reads/{read_id}/Ref_to_signal',
                        data=ref_to_signal)
    file.create_dataset(f'Reads/{read_id}/Reference',
                        data=reference)


def get_read_info_from_file(file_path: t.PathLike, read_id: str) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get signal, ref_to_signal, and reference for a given read_id from a hdf5 file."""
    with h5py.File(file_path, 'r') as hdf5_file:
        return get_read_info_from_open_file(hdf5_file, read_id)


def get_read_info_from_open_file(hdf5_file: h5py.File, read_id: str) -> t.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get signal, ref_to_signal, and reference for a given read_id from an open hdf5 file.
    Returns:
        signal: np.ndarray[int]
        ref_to_signal: np.ndarray[int]
        reference: np.ndarray[int]
    """
    # [()] transforms a DataSet into an ndarray
    signal: np.ndarray = hdf5_file['Reads'][read_id]['Dacs'][()]
    ref_to_signal: np.ndarray = hdf5_file['Reads'][read_id]['Ref_to_signal'][(
    )]
    reference: np.ndarray = hdf5_file['Reads'][read_id]['Reference'][()]

    assert len(ref_to_signal)-1 == len(
        reference), "ref_to_signal must contain exactly one more element than reference"

    return signal, ref_to_signal, reference


def write_data_to_feather_file(folder_path: t.PathLike, data: t.Tuple[np.ndarray, np.ndarray, t.List[int]]) -> None:
    x, y, y_lengths = data
    __make_dir(folder_path)

    feather.write_dataframe(pd.DataFrame(data=list(x)),
                            os.path.join(folder_path, 'x'))
    feather.write_dataframe(pd.DataFrame(data=list(y)),
                            os.path.join(folder_path, 'y'))
    feather.write_dataframe(pd.DataFrame(data=list(y_lengths)),
                            os.path.join(folder_path, 'y_lengths'))


def read_data_from_feather_file(folder_path: t.PathLike) -> t.Tuple[np.ndarray, np.ndarray, t.List[int]]:
    x = feather.read_dataframe(os.path.join(folder_path, 'x'))
    y = feather.read_dataframe(os.path.join(folder_path, 'y'))
    y_lengths = feather.read_dataframe(os.path.join(folder_path, 'y_lengths'))

    return x.to_numpy(), y.to_numpy(dtype=np.float32), y_lengths.to_numpy().flatten().tolist()

def __make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
