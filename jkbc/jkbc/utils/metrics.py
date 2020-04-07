# +
from fastai.basics import *

import jkbc.utils.postprocessing as pop
import jkbc.utils.preprocessing as prep
import jkbc.types as t

from fastai.callbacks.tracker import SaveModelCallback

class Loss():
    '''Abstract class for computing loss'''
    def loss(self):
        pass

class CtcLoss(Loss):
    '''CTC loss'''
    def __init__(self, window_size: int, prediction_size, batch_size: int, alphabet_size:int):
        self.input_to_output_scale = prediction_size/window_size
        self.batch_size = batch_size
        self.alphabet_size = alphabet_size
        self.log_softmax = nn.LogSoftmax(dim=2)
    
    def loss(self) -> functools.partial:
        def __ctc_loss(alphabet_size: int, pred: torch.Tensor, labels: torch.Tensor, pred_lengths, label_lengths) -> float:
            pred = pred.contiguous().view((pred.shape[1], pred.shape[0], alphabet_size))
            
            pred_lengths = (pred_lengths*self.input_to_output_scale)
            return nn.CTCLoss()(self.log_softmax(pred), labels, pred_lengths.int(), label_lengths)
        return partial(__ctc_loss, self.alphabet_size)

class KdLoss(Loss):
    '''Konwledge distillation loss'''
    def __init__(self, alpha: float, temperature: float, label_loss: Loss):
        """Args:
            alpha: How much the teacher controls the training. (1=teacher only, 0=no teacher)
            temperature: Used to smooth output from teacher
            label_loss: Instance of a Loss class
        """
        self.temperature = temperature
        self.label_weight = 1-alpha
        self.teacher_weight = alpha*temperature**2
        self.softmax = nn.Softmax(dim=2)
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.label_loss = label_loss.loss()

    def loss(self) -> functools.partial:
        def __combined(self, y_pred_b: t.Tensor, y_b: t.Tensor, y_lengths: t.List[int]) -> float:
        #def __combined(self, y_pred_b: t.Tensor, y_b: t.Tensor, y_lengths: t.List[int], y_teacher: t.Tensor3D) -> float:
            y_teacher=None
            print(y_lengths)
            label_loss = self.label_loss(self.log_softmax(y_pred_b), y_b, y_lengths)
            teacher = self.__knowledge_distillation_loss(y_pred_b, y_teacher)
            
            return self.label_weight*label+self.teacher_weight*teacher
        
        return partial(__combined, self)
    
    def __knowledge_distillation_loss(self, y_pred_b: t.Tensor, y_teacher: t.Tensor3D) -> float:
        soft_teacher = self.softmax(y_teacher/self.temperature)
        log_soft_pred = self.log_softmax(y_pred_b/self.temperature)
        loss = torch.distributions.kl.kl_divergence(soft_teacher, log_soft_pred)
        return loss

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
        # last_target is a tuple (input_lengths, labels, label_lengths (and y_teacher for KD))
        inp
        labels = last_target[1]
        # Reducing the amount of windows considered
        batch_slice = min(len(last_output), batch_slice)
        x = last_output[:batch_slice].detach()
        # Decode to get predictions
        decoded = pop.decode(x, threshold=threshold, beam_size=beam_size, alphabet=alphabet_str)
        # Getting error for each window
        val, count = 0.0, 0
        for index in range(len(decoded)):
            actual = pop.convert_idx_to_base_sequence(labels[index], alphabet_val)
            accuracy = pop.calc_accuracy(actual, decoded[index])
            val += accuracy
            count += 1
            
        return val, count
    
    assert batch_slice > 0, "batch_slice must be a positive non-zero integer"
    
    alphabet_val = list(alphabet.values())
    alphabet_str = ''.join(alphabet_val)
    return partial(ctc_error, alphabet_val, alphabet_str, beam_size, threshold, batch_slice)
# -


