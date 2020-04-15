import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import jkbc.constants as constants
import jkbc.files.torch_files as f

def get_covariance_diff(teacher, student):
    return np.corrcoef(teacher)-np.corrcoef(student)

def save_figure(data, output, temperature_range):
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(data, vmin=-temperature_range, vmax=temperature_range, cmap='RdBu')
    fig.colorbar(cax)

    # Setting labels to -ACGT
    labels = '#-ACGT' # pyplot acts wierd, so first value is not presented (of by one 1-errors)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.savefig(output)

def get_diff_matrix(teacher, student):
    assert student.shape == teacher.shape, "student and teacher does not have the same size"
    label_count = len(constants.ALPHABET.items())
    diffs = np.zeros((label_count, label_count))

    for t, s in zip(student, teacher):
        diffs += get_covariance_diff(torch.t(s), torch.t(t))
    diffs /= len(student)
    return diffs


# +
def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("predictions_t", help="path to teacher predictions")
    parser.add_argument("predictions_s", help="path to student predictions")
    parser.add_argument("-o", help=f"output path", default='scripts/images/')
    parser.add_argument("-n", help=f"output name. can be used together with path.", default='output.png')
    parser.add_argument("-c", help=f"common path", default='')
    parser.add_argument("-r", help=f"range for image temperature (default=.25)", default=.25)
    args = parser.parse_args()
    
    
    save_path = args.o + args.n
    print("Start comparison")
    teacher = f.load_teacher_data(args.c+args.predictions_t)
    student = f.load_teacher_data(args.c+args.predictions_s)
    
    data = get_diff_matrix(teacher, student)
    
    if not os.path.exists(args.o):
        os.makedirs(args.o)
    save_figure(data, save_path, float(args.r))
    print("Comparison done. Output saved at", save_path)
    
if __name__ == '__main__':
    main()
