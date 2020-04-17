# +
from functools import partial

import torch
# -

# OPTIMIZERS
ADAM_W = 'AdamW'
ADAM = 'Adam'
RMS_PROP = 'RMSProp'



def get_optimizer(config):
    if ADAM_W == config.optimizer:
        return partial(torch.optim.AdamW, amsgrad=True, lr=config.learning_rate)
    elif ADAM == config.optimizer:
        return partial(torch.optim.Adam, amsgrad=True, lr=config.learning_rate, weight_decay=config.weight_decay)
    elif RMS_PROP == config.optimizer:
        return partial(torch.optim.RMSprop, amsgrad=True, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
    else:
        raise NotImplementedError(f"{config.optimizer} optimizer not implemented")
