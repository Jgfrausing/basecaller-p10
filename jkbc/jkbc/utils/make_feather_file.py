import argparse
from pathlib import Path

import numpy as np
import json

import jkbc.types as t
import jkbc.utils.constants as constants
import jkbc.utils.torch_files as f
import jkbc.utils.preprocessing as prep


# DEFAULT PARAMETERS
FOLDERPATH     = 'data/feather-files/'
MAX_WINDOW_SIZE = 4096
MIN_WINDOW_SIZE = 1600
MAX_LABEL_LEN = 400
MIN_LABEL_LEN = 200

def make_file(data_path: t.PathLike, folder_path: t.PathLike, ran: t.List[int], min_label_len: int, max_label_len: int, min_window_size:int, max_window_size:int, blank_id:int, stride:int) -> None:
    # Get data range
    collection = prep.SignalCollection(data_path, labels_per_window=(min_label_len, max_label_len), window_size=(min_window_size, max_window_size), blank_id=blank_id, stride=stride)
    data = collection.get_range(ran);
    
    # Write to file
    f.save_training_data(data, folder_path)

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
    parser.add_argument("--minl", help=f"fixed label length (default {MIN_LABEL_LEN})", default=MIN_LABEL_LEN)
    parser.add_argument("--maxl", help=f"fixed label length (default {MAX_LABEL_LEN})", default=MAX_LABEL_LEN)
    parser.add_argument("--minw", help=f"minimum window size (default {MIN_WINDOW_SIZE})", default=MIN_WINDOW_SIZE)
    parser.add_argument("--maxw", help=f"maximum window size (default {MAX_WINDOW_SIZE})", default=MAX_WINDOW_SIZE)
    parser.add_argument("--s", help=f"stride (default {500})", default=500)
    parser.add_argument("--o", help=f"output path (default '{FOLDERPATH}')", default=FOLDERPATH)
    parser.add_argument("-run_test", help="run validation test after file is saved", action="store_true")

    parser.add_argument("data_path", help="path to data file")
    
    args = parser.parse_args()
    
    base_dir = Path(args.o)
    folder_name = f'Range{args.f}-{args.t}-FixLabelLen{args.maxl}-winsize{args.maxw}'
    output_path = base_dir/folder_name
    print('Making feather files')
    
    data = make_file(args.data_path, output_path, range(int(args.f), int(args.t)),
                     min_label_len=int(args.minl), max_label_len=int(args.maxl), 
                     min_window_size=int(args.minw), max_window_size=int(args.maxw),
                     blank_id=0, stride=int(args.s))
    print('Saving config')
    dump = json.dumps(vars(args))
    with open(output_path/"config.json","w") as f:
        f.write(dump)

    print('Files created')
    

    if args.run_test:
        print('Running test')
        test_read_equal_write(data, output_path)
        print('Test done')

if __name__ == '__main__':
    main()
