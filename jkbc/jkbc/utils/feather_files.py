import argparse
from pathlib import Path

import numpy as np

import jkbc.utils.preprocessing as prep
import jkbc.utils.files as f
import jkbc.types as t


# DEFAULT PARAMETERS
FIX_LABEL_LEN = 60  # Needed to avoid issues with jacked arrays
BLANK_ID = 4
FOLDERPATH = 'data/feather-files/'


def make_new_with_range(data_path: t.PathLike, folder_path: t.PathLike, range_: range, label_len: int, padding_val: int) -> t.Tuple[np.ndarray, np.ndarray]:
    # Get data range
    collection = prep.SignalCollection(data_path)
    data = __get_range(collection, range_, label_len, padding_val)

    # Write to file
    f.write_data_to_feather_file(folder_path, data)

    return data


def __get_range(collection: prep.SignalCollection, range_: range, label_len: int, padding_val: int) -> t.Tuple[np.ndarray, np.ndarray]:
    x = None
    y = None
    for i in range_:
        # Getting data
        data = collection[i]
        data_fields = np.array(data.x), np.array(data.y), data.reference
        _x, _y, _ = data_fields  # we don't use the full reference while training

        # Concating into a single collection
        x = _x if x is None else np.concatenate((x, _x))
        y = _y if y is None else np.concatenate((y, _y))

    # Adding padding
    y_padded = prep.add_label_padding(
        labels=y, fixed_label_len=label_len, padding_val=padding_val)

    return (x, y_padded)


# ## TEST

def test_read_equal_write(data: t.Tuple[np.ndarray, np.ndarray], folder_path: t.PathLike):
    '''Used when changes are made to test that `data > write > read == data`'''
    x, y = data
    x_, y_ = f.read_data_from_feather_file(folder_path)

    assert x_.shape == x.shape
    assert y_.shape == y.shape
    assert x_.dtype == x.dtype
    assert y_.dtype == y.dtype
    __assert_sums_are_equal(x, x_), __assert_sums_are_equal(y, y_)


def __assert_sums_are_equal(a, b):
    assert sum([sum(x) for x in a]) == sum([sum(x) for x in b])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--f", help="range from (default 0)", default=0)
    parser.add_argument("--t", help="range to (default 5)", default=5)
    parser.add_argument(
        "--ll", help=f"fixed label length (default {FIX_LABEL_LEN})", default=FIX_LABEL_LEN)
    parser.add_argument(
        "--b", help=f"blank id (default {BLANK_ID})", default=BLANK_ID)
    parser.add_argument(
        "--o", help=f"output path (default '{FOLDERPATH}')", default=FOLDERPATH)
    parser.add_argument(
        "-run_test", help="run validation test after file is saved", action="store_true")

    parser.add_argument("data_path", help="path to data file")

    args = parser.parse_args()

    base_dir = Path(args.o)
    folder_name = f'Range{args.f}-{args.t}-FixLabelLen{args.ll}'

    output_path = base_dir/folder_name

    print('Making feather files')
    data = make_new_with_range(args.data_path, output_path, range(
        int(args.f), int(args.t)), int(args.ll), int(args.b))
    print('Files created')

    if args.run_test:
        print('Running test')
        test_read_equal_write(data, output_path)
        print('Test done')


if __name__ == '__main__':
    main()
