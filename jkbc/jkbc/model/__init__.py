from fastai.basics import *
import torch
from tqdm import tqdm

import jkbc.types as t
import jkbc.utils.general as g
import jkbc.utils.postprocessing as pop


def get_predicter(model, model_dir, device):
    _empty_tensor = TensorDataset(torch.zeros(1))
    _dl = DataLoader(_empty_tensor)
    databunch = DataBunch(_dl, _dl, device=device)
    return Learner(databunch, model, model_dir=model_dir)


def predict_and_assemble(model, x: t.Tensor, alphabet: str, window_size, stride, beam_size = 25, beam_threshold=0.1) -> t.Tuple[str, t.Tensor2D]:
    # Predict signals
    pred = model(x).detach().cpu()
    # Decode into characters using beam search
    decoded = pop.decode(pred, alphabet, beam_size=beam_size, threshold=beam_threshold)
    # Assemble most likely sequence
    assembled = pop.assemble(decoded, window_size, stride, alphabet)
    
    return assembled, decoded

def get_accuracy(reference: t.List[int], predicted: t.List[str], alphabet_values: t.List[str], return_alignment):
    # Convert reference (list[int]) to string
    actual = pop.convert_idx_to_base_sequence(reference, alphabet_values)
    # Get accuracy
    return pop.calc_accuracy(actual, predicted, return_alignment=return_alignment)

def get_accuracies(refs, seqs, alphabet_values, return_alignment=False):
    accuracy =[]
    for ref, seq in tqdm(zip(refs,seqs)):
        res = get_accuracy(ref, seq, alphabet_values, return_alignment)
        accuracy.append(res)
        
    return accuracy

def predict_range(model, signal_collection, alphabet_string, window_size, device, indexes=None, beam_size=25, beam_threshold=0.1) -> t.List[pop.PredictObject]:
    if indexes is None: 
        indexes = range(len(signal_collection))
        print(f"Warning: predicting all {indexes} signals")
    
    predict_objects = []
    for index in tqdm(indexes):
        # Predict signals
        try:
            read_object = signal_collection[index]
        except:
            print(f'Error: Skipping index {index} (probably because it was empty)')
            continue
        x = signal_to_input_tensor(read_object.x, device)
        assembled, decoded = predict_and_assemble(model, x, alphabet_string, window_size, 1, beam_size, beam_threshold)
        po = pop.PredictObject(read_object.id, decoded, read_object.y, assembled, read_object.reference)
        predict_objects.append(po)
        
    return predict_objects


def load_model_weights(model, model_path, model_name):
    try:
        if not model_name:
            model_name = g.get_newest_model(model_path)
        
        state = torch.load(model_name)
        model = state.pop('model')
        print('Model weights loaded', model_name)
    except:
        print('No model weights available')

def signal_to_input_tensor(signal, device):
    if type(signal) == torch.Tensor:
        x = signal.to(device=device)
    else:
        x = torch.tensor(signal, dtype=torch.float32, device=device)
    
    return x.view(x.shape[0],1,x.shape[1])
