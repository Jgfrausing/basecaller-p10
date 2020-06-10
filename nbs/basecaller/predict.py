#!/usr/bin/env python
# coding: utf-8
# %%
from fastai.basics import *
import json
import wandb
import fire

import jkbc.constants as constants
import jkbc.files.torch_files as f
import jkbc.utils.preprocessing as prep
import jkbc.utils.postprocessing as pop
import jkbc.utils as utils

PROJECT = 'jk-basecalling-v2' 
TEAM="jkbc"
DEVICE = torch.device('cuda')


# %%
def predict(model, data_loader, alphabet: str, beam_size = 25, beam_threshold=0.1):
    # Predict signals
    predictions = []
    labels = []
    model.eval()
    for input, (target, _, _) in iter(data_loader):
        pred = model(input.to('cuda')).detach().cpu()
        decoded = pop.decode(pred, alphabet, beam_size=beam_size, threshold=beam_threshold)
        
        predictions.append(decoded)
        labels.append(target)
    
    return predictions, labels


# %%
def map_decoded(x, y, alphabet_values):
    references = {}
    predictions= {}
    for batch in range(len(x)):
        for index in range(len(x[batch])):
            key = f'#{batch}#{index}#'
            references[key] = pop.convert_idx_to_base_sequence(y[batch][index], alphabet_values)
            predictions[key] = x[batch][index]
    return references, predictions


# %%
def save(model_id, data_name, labels, predictions):
    import jkbc.files.fasta as fasta
    reference_dict, prediction_dict = map_decoded(predictions, labels, list(constants.ALPHABET.values()))
    
    # save fasta
    fasta.save_dicts(prediction_dict, reference_dict, f'predictions/{model_id}-{data_name}')


# %%
def get_config(run_path, root):
    config_path = wandb.restore('config.yaml', run_path=run_path, replace=True, root=root)
    with open(config_path.name, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config


def get_window_size(data_set):
    with open(f'{data_set}/config.json', 'r') as fp:
        data_config = json.load(fp)
        return int(data_config['maxw']) #maxw = max windowsize


# %%
def get_model(config, window_size, device, run_path, root):
    import jkbc.model as m
    import jkbc.model.factory as factory
    import jkbc.utils.bonito.tune as bonito

    # Model
    model_params = utils.get_nested_dict(config, 'model_params')['value']
    model_config = bonito.get_bonito_config(model_params, double_kernel_sizes=False)

    model, _ = factory.bonito(window_size, device, model_config)
    predicter = m.get_predicter(model, device, '')

    weights = wandb.restore('bestmodel.pth', run_path=run_path, replace=True, root=root)
    # fastai requires the name without .pth
    model_weights = '.'.join(weights.name.split('.')[:-1])
    predicter.load(model_weights)
    
    return predicter.model


# %%
def run(id, data_set, data_name='unknow_data_name', batch_size=64):
    run_path = f"{TEAM}/{PROJECT}/{id}"
    root=f'wandb/{id}'
    
    alphabet = ''.join(constants.ALPHABET.values())

    window_size = get_window_size(data_set)
    data = f.load_training_data(data_set) 
    test_dl, _ = prep.convert_to_dataloaders(data, split=.1, batch_size=batch_size, drop_last=True)

    config = get_config(run_path, root)
    model = get_model(config, window_size, DEVICE, run_path, root)

    predictions, labels = predict(model, test_dl, alphabet, 
                                  beam_size = 25, beam_threshold=0.1)

    save(id, data_name, labels, predictions)


# %%
if __name__ == '__main__':
  fire.Fire(run)
