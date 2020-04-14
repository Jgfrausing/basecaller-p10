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

def map_decoded(predict_object, alphabet_values: t.List[str], assembly=False) -> t.Tuple[t.Dict[str, str], t.Dict[str, str]]:
    references = {}
    predictions= {}
    divider = '#'
    for index in tqdm(range(len(predict_object.predictions))):
        key = predict_object.id+divider+str(index)
        references[key] = pop.convert_idx_to_base_sequence(predict_object.references[index], alphabet_values)
        predictions[key] = predict_object.predictions[index]

    if assembly:
        key = predict_object.id+divider+'assembled'
        references[key] = pop.convert_idx_to_base_sequence(predict_object.full_reference, alphabet_values)
        predictions[key] = predict_object.assembled

    return references, predictions


def merge(lst_of_dicts: t.List[t.Dict[str, str]]) -> t.Dict[str, str]:
    return_dict = {}
    for d in lst_of_dicts:
        if bool(set(return_dict) & set(d)):
            raise ValueError('Two dicts share a value')
        return_dict.update(d)
    
    return c.OrderedDict(sorted(return_dict.items())) 


def save_dicts(predictions: t.Dict[str, str], references: t.Dict[str, str], path:str = ''):
    if not os.path.exists(path):
        os.makedirs(path)
    save_fasta_file(predictions, f'{path}/prediction.fasta')
    save_fasta_file(references, f'{path}/reference.fasta')
