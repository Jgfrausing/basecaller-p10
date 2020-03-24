# +
from fastai.basics import *

import jkbc.utils.postprocessing as pop
import jkbc.utils.preprocessing as prep
import jkbc.types as t


# -

def ctc_loss(prediction_size: int, batch_size: int, alphabet_size:int) -> functools.partial:
    def __ctc_loss(y_pred_lengths, alphabet_size, y_pred_b: torch.Tensor, y_b: torch.Tensor, y_lengths) -> float:
        if y_pred_lengths.shape[0] != y_pred_b.shape[0]:
            new_len = y_pred_b.shape[0]
            y_pred_lengths_ = y_pred_lengths[:new_len]
        else:
            y_pred_lengths_ = y_pred_lengths

        y_pred_b_ = y_pred_b.view((y_pred_b.shape[1], y_pred_b.shape[0], alphabet_size))

        return nn.CTCLoss()(y_pred_b_, y_b, y_pred_lengths_, y_lengths)
    return partial(__ctc_loss, prep.get_prediction_lengths(prediction_size, batch_size), alphabet_size)

class ErrorRate(Callback):
    "Error rate metrics computation."
    def __init__(self, error_function):
        """Args:
            error_function: function that accepts last_output, last_target, and **kwargs
        """
        self.error_function = error_function
        self.name = getattr(error_function,'func',error_function).__name__
        
    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = .0, 0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        val, count = self.error_function(last_output, last_target)
        self.val+=val
        self.count+=count
            
    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val/self.count)


def ctc_error(alphabet:t.Dict[int, str], beam_size:int = 2, threshold:int =.0, batch_slice:int = 5) -> functools.partial:
    """CTC accuracy function to use with ErrorRate.

    Args:
        alphabet: Dictionary from integer to character including blank
        beam_size: the number of candidates to consider
        threshold: characters below this threshold are not considered
        batch_slice: How many windows to consider within each batch
    Returns:
        Average Rates.error for the considered windows
    """
    def ctc_error(alphabet_val, alphabet_str, beam_size, threshold, batch_slice, last_output, last_target, **kwargs):
        # last_target is a tuple (labels, label_lengths)
        labels = last_target[0]
        # Reducing the amount of windows considered
        batch_slice = min(len(last_output), batch_slice)
        x = last_output.detach().cpu().numpy()[:batch_slice]
        # Decode to get predictions
        decoded = pop.decode(x, threshold=threshold, beam_size=beam_size, alphabet=alphabet_str)
        # Getting error for each window
        val, count = 0.0, 0
        for index in range(len(decoded)):
            actual = pop.convert_idx_to_base_sequence(labels[index], alphabet_val)
            error = pop.calc_sequence_error_metrics(actual, decoded[index])
            val += error.error
            count += 1
            
        return val, count
    
    assert batch_slice > 0, "batch_slice must be a positive non-zero integer"
    
    alphabet_val = list(alphabet.values())
    alphabet_str = ''.join(alphabet_val)
    return partial(ctc_error, alphabet_val, alphabet_str, beam_size, threshold, batch_slice)
