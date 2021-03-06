# +
import math

from fastai.basics import *
from fastai.callbacks.tracker import SaveModelCallback, EarlyStoppingCallback
from fastai.callbacks import CSVLogger
import numpy as np
import torch.distributions as dist
# -

import jkbc.utils.postprocessing as pop
import jkbc.utils.preprocessing as prep
import jkbc.types as t

class Loss():
    '''Abstract class for computing loss'''
    def loss(self):
        pass

class CtcMetric(Callback):
    def __init__(self):
        self.name = 'ctc_loss'
        self.loss = 0
        self.count = 0
        self.can_set = False
        
    def reset_loss(self):
        #self.loss, self.count = 0, 0
        self.can_set = False
        
    def set_loss(self, loss):
        if self.can_set:
            self.loss += loss
            self.count += 1
        
    def on_batch_begin(self, **kwargs):
        # If we are not training (e.i we are validating), we allow set_loss
        self.can_set = not kwargs['train'] if 'train' in kwargs else False
        
    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        mean_loss = self.loss/self.count if self.count != 0 else 0
        self.loss = 0
        self.count = 0
        self.can_set = False
            
        return add_metrics(last_metrics, mean_loss.cpu())


class CtcLoss(Loss):
    '''CTC loss'''
    def __init__(self, prediction_scale, batch_size: int, alphabet_size:int):
        self.input_to_output_scale = prediction_scale
        self.batch_size = batch_size
        self.alphabet_size = alphabet_size
        self.log_softmax = nn.LogSoftmax(dim=2)
        self.metric = CtcMetric()
    
    def loss(self) -> functools.partial:
        def ctc_loss(alphabet_size: int, metric: CtcMetric, *tensors) -> float: #pred: torch.Tensor, labels: torch.Tensor, pred_lengths, label_lengths) -> float:
            if len(tensors) == 4:
                pred, labels, pred_lengths, label_lengths = tensors
            else:
                pred, labels, pred_lengths, label_lengths, _ = tensors
            pred_lengths = pred_lengths/self.input_to_output_scale
            
            loss = nn.CTCLoss()(self.log_softmax(pred).transpose(1,0), labels, pred_lengths, label_lengths)
            metric.set_loss(loss)
            return loss
        return partial(ctc_loss, self.alphabet_size, self.metric)

class KdLoss(Loss):
    '''Konwledge distillation loss'''
    def __init__(self, alpha: float, temperature: float, label_loss: Loss, variant='logsoftmax'):
        """Args:
            alpha: How much the teacher controls the training. (1=teacher only, 0=no teacher)
            temperature: Used to smooth output from teacher
            label_loss: Instance of a Loss class
            variant: 
                - logsoftmax (or True): logsoftmax on teacher output
                - softmax: softmax on teacher output
        """
    
        def get_softmax(variant):
            if 'logsoftmax' == variant or variant is True:
                return nn.LogSoftmax(dim=2)
            elif 'softmax' == variant:
                return nn.Softmax(dim=2)
        self.temperature = temperature
        self.label_weight = 1-alpha
        self.teacher_weight = alpha*temperature**2
        self.teacher_softmax = get_softmax('softmax')
        self.student_softmax = get_softmax(variant)
        self.label_loss = label_loss.loss()

    def loss(self) -> functools.partial:
        def __combined(self, pred: torch.Tensor, labels: torch.Tensor, pred_lengths, label_lengths, y_teacher: t.Tensor3D) -> float:
            # requires: pred, labels, pred_lengths, label_lengths)
            label_loss = self.label_loss(pred, labels, pred_lengths, label_lengths)
            
            teacher_loss = self.__knowledge_distillation_loss(pred, y_teacher)
            loss = self.label_weight*label_loss+self.teacher_weight*(teacher_loss/10000)
            return loss
        
        return partial(__combined, self)
    
    def __knowledge_distillation_loss(self, y_pred_b: t.Tensor, y_teacher: t.Tensor3D) -> float:
        soft_teacher = self.teacher_softmax(y_teacher/self.temperature)
        soft_pred = self.student_softmax(y_pred_b/self.temperature)
        loss = nn.KLDivLoss(reduction='batchmean')(soft_pred, soft_teacher)
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


def read_identity(alphabet:t.Dict[int, str], beam_size:int = 2, threshold:int =.0, batch_slice:int = 20) -> functools.partial:
    """CTC accuracy function to use with ErrorRate.

    Args:
        alphabet: Dictionary from integer to character including blank
        beam_size: the number of candidates to consider
        threshold: characters below this threshold are not considered
        batch_slice: How many windows to consider within each batch
    Returns:
        Average Rates.error for the considered windows
    """
    def read_identity(alphabet_val, alphabet_str, beam_size, threshold, batch_slice, last_output, last_target, **kwargs):
        # last_target is a tuple of (y, x_lengths, y_lengths, and maybe a teacher))
        labels = last_target[0]
        
        # Reducing the amount of windows considered
        batch_slice = min(len(last_output), batch_slice)
        x = last_output[:batch_slice].detach()
        # Decode to get predictions
        decoded = pop.decode(x, threshold=threshold, beam_size=beam_size, alphabet=alphabet_str)
        # Getting error for each window
        val, count = 0.0, 0
        for index in range(len(decoded)):
            actual = pop.convert_idx_to_base_sequence(labels[index], alphabet_val)
            accuracy = pop.calc_accuracy(actual, decoded[index], return_alignment=False)
            val += accuracy
            count += 1
            
        return val, count
    
    assert batch_slice > 0, "batch_slice must be a positive non-zero integer"
    
    alphabet_val = list(alphabet.values())
    alphabet_str = ''.join(alphabet_val)
    return ErrorRate(partial(read_identity, alphabet_val, alphabet_str, beam_size, threshold, batch_slice))
