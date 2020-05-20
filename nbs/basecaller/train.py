# +
import yaml
import json

from fastai.basics import *
from fastai.callbacks.tracker import EarlyStoppingCallback
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

# Optimal setting for fast model predictions
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic= False

# +
BASE_DIR = Path("../..")
PATH_DATA = 'data/feather-files'
DATA_SET = BASE_DIR/PATH_DATA/'Range0-10000-FixLabelLen400-winsize4096'
DATA_SET_SMALL = BASE_DIR/PATH_DATA/'Range0-1000-FixLabelLen400-winsize4096'
PROJECT_V1 = 'jk-basecalling' 
PROJECT_V2= 'jk-basecalling-v2'
TEAM="jkbc"
DEFAULT_CONFIG = 'config_default.yaml'
DEFAULT_CONFIG_MODIFIED = 'config_default_modified.yaml'

DEVICE = torch.device("cuda") #m.get_available_gpu()
ALPHABET = constants.ALPHABET
ALPHABET_SIZE = len(ALPHABET.values())


# -

def run(data_set=DATA_SET, id=None, epochs=20, new=False, device=DEVICE, batch_size=340, config=DEFAULT_CONFIG, kd_method=None, tags=[], project=PROJECT_V1):
    PROJECT = project
    PROJECT_PATH = f'{TEAM}/{PROJECT}'
    
    # Load default dictionary
    if type(config) is not dict:
        with open(config, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    if kd_method is not None:
        config['knowledge_distillation'] = kd_method
    
    if False == config['knowledge_distillation']:
        # setting alpha and temperature to none to avoid confusion
        config['kd_alpha'] = None
        config['kd_temperature'] = None

    if id and not new: # Resume run from wandb
        wandb.init(config=config, resume='allow', id=id, entity=TEAM, project=PROJECT, tags=tags, reinit=True)
    else:              # Start new run
        wandb.init(config=config, resume='allow', entity=TEAM, project=PROJECT, tags=tags, reinit=True)
    
    # Use sweep to restart multiple existing runs using ids
    if 'id' in wandb.config.keys():
        new = True
        id = wandb.config['id']
    
    if new:
        # Load config from run specified
        config = wandb.restore('config.yaml', run_path=f"{PROJECT_PATH}/{id}", replace=True)
        with open(config.name, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)

        # Clean up dictionary
        config = {k: v['value'] for k, v in config.items() if k not in ['wandb_version', '_wandb']}
        wandb.config.update(config, allow_val_change=True)
        wandb.config['started_from'] =  id
        wandb.run.save()

    config = wandb.config
    
    print(f'Dataset: {data_set}\nResumed: {wandb.run.resumed}\nId: {wandb.run.id}\nDevice: {device}')

    with open(f'{data_set}/config.json', 'r') as fp:
        data_config = json.load(fp)
        window_size = int(data_config['maxw']) #maxw = max windowsize

    # Get model
    model = __get_model(config, window_size, device)
    config.parameters = m.get_parameter_count(model)
    print('Parameters:', config.parameters)
    
    config.time_predict = m.time_model_prediction(model, device)
    print('Time:', config.time_predict)
    wandb.run.save()
    
    if config.max_parameters is not None and config.max_parameters < config.parameters:
        raise ValueError(f"Too many parameters ({config.parameters})")

    # Loss, metrics and callback
    _ctc_loss = metric.CtcLoss(config.dimensions_out_scale, batch_size, ALPHABET_SIZE)
    if config.knowledge_distillation:
        _kd_loss = metric.KdLoss(alpha=config.kd_alpha, temperature=config.kd_temperature, label_loss=_ctc_loss, variant=config.knowledge_distillation)
        loss = _kd_loss.loss()
    else:
        loss = _ctc_loss.loss()

    metrics = [metric.read_identity(ALPHABET, 5)]

    # Load data   
    databunch = __load_data(config, data_set, device, batch_size)
    
    # training
    optimizer = optim.get_optimizer(config)
    scheduler = sched.get_scheduler(config, epochs)
    early_stopper = partial(EarlyStoppingCallback, monitor='valid_loss', min_delta=0.01, patience=3)
    callbacks = [early_stopper, WandbCallback]
    
    # Learner
    learner = Learner(databunch, model, loss_func=loss, metrics=metrics, opt_func=optimizer, callback_fns=callbacks)
    scheduler(learner)

    if wandb.run.resumed or new:
        try:
            weights = wandb.restore('bestmodel.pth', run_path=f"{PROJECT_PATH}/{id}")
            # fastai requests the name without .pth
            weight_name = '.'.join(weights.name.split('.')[:-1])
            learner.load(weight_name)
            print(f'Loaded best model from id {id}')
        except:
            print('No model found')

    print('Epochs:', epochs)
    learner.fit(epochs, lr=config.learning_rate, wd=config.weight_decay)


def run_modified_configs(function_identifier, original_config=DEFAULT_CONFIG_MODIFIED, data_set=DATA_SET_SMALL, tags=[]):    
    with open(original_config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    if type(tags) is not list:
        tags = [tags]
        
    configs, t = factory.modify_config(function_identifier, config)
    for c in configs:
        tags += t
        try:
            run(data_set=data_set, id=None, epochs=30, batch_size=64, config=c, tags=tags, project=PROJECT_V2)
            wandb.join()
        except Exception as e:
            print('config:', c)
            print(e)


def print_configs(function_identifier, original_config=DEFAULT_CONFIG_MODIFIED):
    with open(original_config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
        
    configs, t = factory.modify_config(function_identifier, config)
    for c in configs:
        print(c)


def __load_data(config, data_set, device, batch_size):
    if config.knowledge_distillation:
        data, teacher = f.load_training_data_with_teacher(data_set, config.teacher_name)
        train_dl, valid_dl = prep.convert_to_dataloaders(data, split=.8, batch_size=batch_size, teacher=teacher, drop_last=config.drop_last)
    else:
        data = f.load_training_data(data_set) 
        train_dl, valid_dl = prep.convert_to_dataloaders(data, split=.8, batch_size=batch_size, drop_last=config.drop_last)

    # Convert to databunch
    return DataBunch(train_dl, valid_dl, device=device)


def __get_model(config, window_size, device):
    model_params = utils.get_nested_dict(config, 'model_params')
    config.update({'model_params': model_params}, allow_val_change=True)
    model_config = bonito.get_bonito_config(config.model_params)
    model, config.dimensions_out_scale = factory.bonito(window_size, device, model_config)

    return model


if __name__ == '__main__':
    run()
