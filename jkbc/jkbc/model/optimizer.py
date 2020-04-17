# +
from functools import partial

import torch
# -

# OPTIMIZERS
ADAM_W = 'AdamW'


def get_optimizer(config):
    if ADAM_W == config.optimizer:
        return partial(torch.optim.AdamW, amsgrad=True, lr=config.learning_rate)
    else:
        raise NotImplementedError(f"{config.optimizer} optimizer not implemented")
