import jkbc.types as t
import jkbc.utils.postprocessing as pop

def get_stats(prediction: t.Tensor2D, actual: str, alphabet: t.List[str], beam_sizes: t.List[int]):
    print(actual)
    y_pred_index = prediction[None,:,:]
    for beam in beam_sizes:
        decoded = pop.decode(y_pred_index, threshold=.0, beam_size=beam, alphabet=alphabet)   
        predicted = decoded[0]
        error = pop.calc_sequence_error_metrics(actual, predicted)
        yield (predicted, beam, error)