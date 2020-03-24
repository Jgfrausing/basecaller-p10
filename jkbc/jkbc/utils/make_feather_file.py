import argparse
from pathlib import Path

import numpy as np

import jkbc.utils.preprocessing as prep
import jkbc.utils.files as f
import jkbc.types as t


# DEFAULT PARAMETERS
FIX_LABEL_LEN  = 70  # Needed to avoid issues with jacked arrays
BLANK_ID       = prep.BLANK_ID
FOLDERPATH     = 'data/feather-files/'
STRIDE         = 300 # We make stride same size as window to make more distinct training data

def make_file(data_path: t.PathLike, folder_path: t.PathLike, ran: range, label_len: int) -> None:
    # Get data range
    collection = prep.SignalCollection(data_path, max_labels_per_window=FIX_LABEL_LEN, stride=STRIDE)
    data = collection.get_range(ran, label_len);

    # Write to file
    f.write_data_to_feather_file(folder_path, data)

    return data

# ## TEST

# Used when changes are made to test that `data > write > read == data`
def test_read_equal_write(data: t.Tuple[np.ndarray, np.ndarray], folder_path: t.PathLike):
    x , y,  y_lengths = data
    x_, y_, y_lengths_ = f.read_data_from_feather_file(folder_path)


    assert x_.shape == x.shape
    assert y_.shape == y.shape
    assert len(y_lengths_) == len(y_lengths)
    assert x_.dtype == x.dtype
    assert y_.dtype == y.dtype
    assert type(y_lengths_[0]) == type(y_lengths[0]), f'{type(y_lengths_[0])} <> {type(y_lengths[0])}'
    _equal_sum(x, x_), _equal_sum(y, y_)
    assert sum(y_lengths) == sum(y_lengths_)

def _equal_sum(a, b):
    assert sum([sum(x) for x in a]) == sum([sum(x) for x in b])


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--f", help="range from (default 0)", default=0)
    parser.add_argument("--t", help="range to (default 5)", default=5)
    parser.add_argument("--ll", help=f"fixed label length (default {FIX_LABEL_LEN})", default=FIX_LABEL_LEN)
    parser.add_argument("--o", help=f"output path (default '{FOLDERPATH}')", default=FOLDERPATH)
    parser.add_argument("-run_test", help="run validation test after file is saved", action="store_true")

    parser.add_argument("data_path", help="path to data file")
    
    args = parser.parse_args()
    
    base_dir = Path(args.o)
    folder_name = f'Range{args.f}-{args.t}-FixLabelLen{args.ll}'
    output_path = base_dir/folder_name
    print('Making feather files')
    data = make_file(args.data_path, output_path, range(int(args.f), int(args.t)), int(args.ll))
    print('Files created')

    if args.run_test:
        print('Running test')
        test_read_equal_write(data, output_path)
        print('Test done')


if __name__ == '__main__':
    main()
