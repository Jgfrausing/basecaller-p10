import jkbc.types as t
import jkbc.utils.general as g
import jkbc.utils.postprocessing as pop


def predict(learner, x: t.Tensor, alphabet: t.Dict[int, str], window_size, stride, reference: str=None, beam_size = 25, beam_threshold=0.1) -> t.Tuple[str, t.Tuple[float, str]]:
    alphabet_values = list(alphabet.values())
    alphabet_str = ''.join(alphabet_values)

    # Predict signals
    pred = learner.pred_batch(x).detach()
    # Decode into characters using beam search
    decoded = pop.decode(pred, alphabet_str, beam_size=beam_size, threshold=beam_threshold)
    # Assemble most likely sequence
    assembled = pop.assemble(decoded, window_size, stride, alphabet)
    
    # Return assembled, if no reference is available for comparison
    if reference is None: 
        return assembled, None
    
    # Convert reference (list[int]) to string
    actual = pop.convert_idx_to_base_sequence(reference, alphabet_values)
    # Get accuracy
    accuracy, alignment = pop.calc_accuracy(actual, assembled, return_alignment=True)
    return assembled, (accuracy, alignment)


def load_model_weights(learner, model_name):
    try:
        if not model_name:
            model_name = g.get_newest_model(learner.model_dir)
            
        learner = learner.load(model_name)
        print('Model weights loaded', model_name)
    except:
        print('No model weights available')
