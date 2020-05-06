# +
import yaml
import json

from fastai.basics import *
import wandb
from wandb.fastai import WandbCallback
# -

import jkbc.constants as constants
import jkbc.model as m
import jkbc.model.factory as factory
import jkbc.model.optimizer as optim
import jkbc.model.scheduler as sched
import jkbc.files.torch_files as f
import jkbc.model.metrics as metric
import jkbc.utils.preprocessing as prep
import jkbc.utils.bonito.tune as bonito
import jkbc.utils as utils

# +
BASE_DIR = Path("../..")
PATH_DATA = 'data/feather-files'
DATA_SET = BASE_DIR/PATH_DATA/'Range0-10000-FixLabelLen400-winsize4096'
PROJECT = 'jk-basecalling'
TEAM="jkbc"
PROJECT_PATH = f'{TEAM}/{PROJECT}'
DEFAULT_CONFIG = 'config_default.yaml'

DEVICE = torch.device("cuda")
ALPHABET = constants.ALPHABET
ALPHABET_SIZE = len(ALPHABET.values())


# -

def run(data_set=DATA_SET, name='bonito', epochs=40, device=DEVICE, batch_size=128, config=DEFAULT_CONFIG):
    # Load default dictionary
    with open(config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    config['knowledge_distillation'] = False
    config['kd_alpha'] = None
    config['kd_temperature'] = None
    wandb.init(config=config, resume='allow', name=name, entity=TEAM, project=PROJECT)
    config = wandb.config
    
    with open(f'{data_set}/config.json', 'r') as fp:
        data_config = json.load(fp)
        window_size = int(data_config['maxw']) #maxw = max windowsize

    # Get model
    model = __get_model(config, None, window_size, device)
    parameters = m.get_parameter_count(model)
    print('Parameters:', parameters)
    try:
        config.parameters = parameters
    except:
        pass # Already set
    
    time_predict = m.time_model_prediction(model, device)
    print('Time:', time_predict)
    try:
        config.time_predict = time_predict
    except:
        pass # Already set
    
    # Loss, metrics and callback
    _ctc_loss = metric.CtcLoss(config.dimensions_out_scale, batch_size, ALPHABET_SIZE)
    loss = _ctc_loss.loss()

    metrics = [metric.read_identity(ALPHABET, 5)]
    
    # Load data   
    databunch = __load_data(config, data_set, device, batch_size)
    # Learner
    optimizer = optim.get_optimizer(config)
    scheduler = sched.get_scheduler(config, epochs)

    learner = Learner(databunch, model, loss_func=loss, metrics=metrics, opt_func=optimizer, callback_fns=WandbCallback).to_fp16()
    scheduler(learner)

    print('Epochs:', epochs)
    learner.fit(epochs, lr=config.learning_rate, wd=config.weight_decay)


def __load_data(config, data_set, device, batch_size):

    data = f.load_training_data(data_set) 
    train_dl, valid_dl = prep.convert_to_dataloaders(data, split=.8, batch_size=batch_size, drop_last=config.drop_last)

    # Convert to databunch
    return DataBunch(train_dl, valid_dl, device=device)


def __get_model(config, scale_output_to_size, window_size, device):
    config['model_params']['scale_output_to_size'] = None
    model_config = bonito.get_bonito_config(config.model_params)
    model, config.dimensions_out_scale = factory.bonito(window_size, device, model_config)

    return model


if __name__ == '__main__':
    run()
