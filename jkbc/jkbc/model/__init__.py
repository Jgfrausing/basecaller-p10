# +
import os
import toml

from fastai.basics import *
import torch
from tqdm import tqdm
# -

import jkbc.types as t
import jkbc.utils.postprocessing as pop


def get_predicter(model, device, root_dir):
    _empty_tensor = TensorDataset(torch.zeros(1))
    _dl = DataLoader(_empty_tensor)
    databunch = DataBunch(_dl, _dl, device=device)
    return Learner(databunch, model, model_dir=root_dir)


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
    if indexes is None or len(indexes) > len(signal_collention): 
        indexes = range(len(signal_collection))
        print(f"Warning: predicting all {indexes} signals")
            
    predict_objects = []
    for index in tqdm(indexes):
        # Predict signals
        try:
            read_object = signal_collection[index]
        except Exception as e:
            print(f'Error: Skipping index {index}. {e}')
            continue
        x = signal_to_input_tensor(read_object.x, device)
        assembled, decoded = predict_and_assemble(model, x, alphabet_string, window_size, 1, beam_size, beam_threshold)
        po = pop.PredictObject(read_object.id, read_object.bacteria, decoded, read_object.y, assembled, read_object.reference)
        predict_objects.append(po)
        
    return predict_objects


def load_model_weights(model, model_path, model_name):
    try:
        if not model_name:
            model_name = get_newest_model(model_path)
        
        state = torch.load(model_name)
        model = state.pop('model')
        print('Model weights loaded', model_name)
    except:
        print('No model weights available')

def signal_to_input_tensor(signal, device):
    if type(signal) == torch.Tensor:
        x = signal.to(device=device, dtype=torch.float16)
    else:
        x = torch.tensor(signal, dtype=torch.float16, device=device)
    
    return x.view(x.shape[0],1,x.shape[1])


# +
def get_newest_model(folder_path: t.PathLike):
    import glob
    list_of_files = glob.glob(os.path.join(folder_path, '*')) # * means all if need specific format then *.csv
    if len(list_of_files) == 0: return None
    return __get_file_name(max(list_of_files, key=os.path.getctime))

def __get_file_name(file_path: t.PathLike):
    file_name, extension = os.path.splitext(os.path.basename(file_path))

    return file_name


# -

def get_available_gpu():
    ## https://discuss.pytorch.org/t/how-can-find-available-gpu/24281/2
    tmp_file = 'c53c4dc67fd811eabc550242ac130003'
    os.system(f'nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >{tmp_file}')
    memory_available = [int(x.split()[2]) for x in open(tmp_file, 'r').readlines()]
    os.system(f'rm -f {tmp_file}')
    gpu = f'cuda:{np.argmax(memory_available)}'
    return torch.device(gpu)


def save_setup(model_dir, model, data_set, knowlegde_distillation, teacher, batch_size, device):
    os.makedirs(model_dir)
    obj = {'model': model, 'data_set': data_set, 'knowlegde_distillation': knowlegde_distillation,
            'teacher': teacher, 'batch_size': batch_size, 'device': device}
    
    toml_string = toml.dumps(obj)
    with open(f'{model_dir}/definition.toml', 'w') as f:
        f.writelines(toml_string)


def get_parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def time_model_prediction(model, device, count=100):
    ## count increased to 100 from 20 to increase accuracy.
    ## we device by 5 to make the measure backwards compatible 
    import timeit
    signal = torch.ones(128, 4096, device=device)
    input = signal_to_input_tensor(signal, device)
    fn = lambda: model(input).detach().cpu()
    return timeit.timeit(fn, number=count)/5
