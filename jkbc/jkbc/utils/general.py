import os

import jkbc.types as t
import jkbc.utils.postprocessing as pop


def get_newest_model(folder_path: t.PathLike):
    import glob
    list_of_files = glob.glob(os.path.join(folder_path, '*')) # * means all if need specific format then *.csv
    
    return max(list_of_files, key=os.path.getctime)


def __get_file_name(file_path: t.PathLike):
    file_name, extension = os.path.splitext(os.path.basename(file_path))

    return file_name, extension

