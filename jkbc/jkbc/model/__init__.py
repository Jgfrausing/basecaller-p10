import torch

import jkbc.types as t
import jkbc.utils.general as g
import jkbc.utils.postprocessing as pop


def predict_and_assemble(model, x: t.Tensor, alphabet: str, window_size, stride, beam_size = 25, beam_threshold=0.1) -> t.Tuple[str, t.Tensor2D]:
    # Predict signals
    pred = model(x).detach().cpu()
    # Decode into characters using beam search
    decoded = pop.decode(pred, alphabet, beam_size=beam_size, threshold=beam_threshold)
    # Assemble most likely sequence
    assembled = pop.assemble(decoded, window_size, stride, alphabet)
    
    return assembled, decoded
    
def get_accuracy(reference: t.List[int], predicted: t.List[str], alphabet_values: t.List[str]):
    # Convert reference (list[int]) to string
    actual = pop.convert_idx_to_base_sequence(reference, alphabet_values)
    # Get accuracy
    return pop.calc_accuracy(actual, predicted, return_alignment=True)

def load_model_weights(learner, model_name):
    try:
        if not model_name:
            model_name = g.get_newest_model(learner.model_dir)
            
        learner = learner.load(model_name)
        print('Model weights loaded', model_name)
    except:
        print('No model weights available')

def signal_to_input_tensor(signal, device):
    x = torch.tensor(signal, dtype=torch.float32, device=device)
    return x.view(x.shape[0],1,x.shape[1])
