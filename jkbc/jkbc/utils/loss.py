from fastai.basics import *
import jkbc.utils.preprocessing as prep

def ctc_loss(prediction_size: int, batch_size: int, alphabet_size:int) -> functools.partial:
    return partial(__ctc_loss, prep.get_prediction_lengths(prediction_size, batch_size), alphabet_size)

def __ctc_loss(y_pred_lengths, alphabet_size, y_pred_b: torch.Tensor, y_b: torch.Tensor, y_lengths) -> float:
    if y_pred_lengths.shape[0] != y_pred_b.shape[0]:
        new_len = y_pred_b.shape[0]
        y_pred_lengths_ = y_pred_lengths[:new_len]
    else:
        y_pred_lengths_ = y_pred_lengths
        
    y_pred_b_ = y_pred_b.view((y_pred_b.shape[1], y_pred_b.shape[0], alphabet_size))

    return nn.CTCLoss()(y_pred_b_, y_b, y_pred_lengths_, y_lengths)
