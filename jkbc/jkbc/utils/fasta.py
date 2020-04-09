import collections as c
import os
import re

from tqdm import tqdm

import jkbc.types as t
import jkbc.utils.postprocessing as pop

def save_fasta_file(entities: t.Dict[str, str], filename: t.PathLike)->None:
    ordered = c.OrderedDict(sorted(entities.items()))
    with open(filename, 'w') as f:
        for name, seq in ordered.items():
            f.write(f">{name}\n")
            seq = re.sub("(.{80})", "\\1\n", seq, 0, re.DOTALL) #Adding newlines
            f.write(f"{seq}\n")

def save_preditions_to_fasta(read_object, decoded, path: t.PathLike, alphabet_values: t.List[str], assembly=None) -> None:
    references = {}
    predictions= {}
    for index in tqdm(range(len(decoded))):
        references[str(index)] = pop.convert_idx_to_base_sequence(read_object.y[index], alphabet_values)
        predictions[str(index)] = decoded[index]

    if assembly:
        references['assembled'] = pop.convert_idx_to_base_sequence(read_object.reference, alphabet_values)
        predictions['assembled'] = assembly

    if not os.path.exists(path):
        os.makedirs(path)

    save_fasta_file(references, f'{path}/{read_object.id}.reference.fasta')
    save_fasta_file(predictions, f'{path}/{read_object.id}.prediciton.fasta')