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

BASE_DIR = Path("../..")
PATH_DATA = 'data/feather-files'
DATA_SET = BASE_DIR/PATH_DATA/'Range0-500-FixLabelLen400-winsize4096'
PROJECT = 'jk-basecalling'
TEAM="jkbc"
PROJECT_PATH = f'{TEAM}/{PROJECT}'
DEFAULT_CONFIG = 'config_default.yaml'
DEVICE = m.get_available_gpu()


def run(data_set=DATA_SET, id=None, scale_output_to_size=None, epochs=10, new=False, device=DEVICE, batch_size=340, config=DEFAULT_CONFIG):
    # Load default dictionary
    with open(config, 'r') as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    if id and not new: # Resume run from wandb
        wandb.init(config=config, resume='allow', id=id, entity=TEAM, project=PROJECT)
    else:              # Start new run
        wandb.init(config=config, resume='allow', entity=TEAM, project=PROJECT)

    if new:
        # Load config from run specified
        config = wandb.restore('config.yaml', run_path=f"{PROJECT_PATH}/{id}", replace=True)
        with open(config.name, 'r') as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)

        # Clean up dictionary
        config = {k: v['value'] for k, v in config.items() if k not in ['wandb_version', '_wandb']}
        wandb.config.update(config, allow_val_change=True)
        wandb.run.save()

    config = wandb.config
    
    print(f'Dataset: {data_set}\nResumed: {wandb.run.resumed}\nId: {wandb.run.id}\nDevice: {device}')

    with open(f'{data_set}/config.json', 'r') as fp:
        data_config = json.load(fp)
        window_size = int(data_config['maxw']) #maxw = max windowsize

    ALPHABET          = constants.ALPHABET
    ALPHABET_SIZE     = len(ALPHABET.values())

    # Get model
    model = __get_model(config, scale_output_to_size, window_size, device)
    config.parameters = m.get_parameter_count(model)
    print('Parameters:', config.parameters)
    wandb.run.save()
    
    if config.max_parameters < config.parameters:
        raise ValueError(f"Too many parameters ({config.parameters})")

    # Loss, metrics and callback
    _ctc_loss = metric.CtcLoss(config.dimensions_out_scale, batch_size, ALPHABET_SIZE)
    _kd_loss = metric.KdLoss(alpha=config.kd_alpha, temperature=config.kd_temperature, label_loss=_ctc_loss)
    loss = _kd_loss.loss() if config.knowledge_distillation else _ctc_loss.loss()

    metrics = [metric.read_identity(ALPHABET, 5)]

    # Load data   
    databunch = __load_data(config, data_set, device, batch_size)
    
    # Learner
    optimizer = optim.get_optimizer(config)
    scheduler = sched.get_scheduler(config, epochs)

    learner = Learner(databunch, model, loss_func=loss, metrics=metrics, opt_func=optimizer, callback_fns=WandbCallback).to_fp16()
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


def __load_data(config, data_set, device, batch_size):
    if config.knowledge_distillation:
        data, teacher = f.load_training_data_with_teacher(data_set, config.teacher_name)
        train_dl, valid_dl = prep.convert_to_dataloaders(data, split=.8, batch_size=batch_size, teacher=teacher, drop_last=config.drop_last)
    else:
        data = f.load_training_data(data_set) 
        train_dl, valid_dl = prep.convert_to_dataloaders(data, split=.8, batch_size=batch_size, drop_last=config.drop_last)

    # Convert to databunch
    return DataBunch(train_dl, valid_dl, device=device)


def __get_model(config, scale_output_to_size, window_size, device):
    config['model_params']['scale_output_to_size'] = int(scale_output_to_size) if scale_output_to_size else None
    model_params = utils.get_nested_dict(config, 'model_params')
    config.update({'model_params': model_params}, allow_val_change=True)
    model_config = bonito.get_bonito_config(config.model_params)
    model, config.dimensions_out_scale = factory.bonito(window_size, device, model_config)

    return model


if __name__ == '__main__':
    run()
