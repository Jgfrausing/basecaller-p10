from fastai.basics import *
import json
import toml
from tqdm import tqdm
import wandb
from wandb.fastai import WandbCallback

import jkbc.model as m
import jkbc.model.factory as factory
import jkbc.model.optimizer as optimizer
import jkbc.model.scheduler as scheduler
import jkbc.constants as constants
import jkbc.files.torch_files as f
import jkbc.model.metrics as metric
import jkbc.utils.preprocessing as prep
import jkbc.utils.postprocessing as pop
import jkbc.files.fasta as fasta

def main():
    DEVICE = m.get_available_gpu() 
    print("Device:", DEVICE)

    config = dict(
        ## Device
        device = DEVICE,
        
        # Data
        data_set = 'Range0-5-FixLabelLen400-winsize4096',
        alphabet = constants.ALPHABET,
        
        # Training
        teacher_name = 'bonito',
        knowledge_distillation = True,
        pretrained_weights = None,
        epochs = 25,
        batch_size = 2**6,
        learning_rate = 0.001,
        learning_rate_min = 0,
        dropout = 0.1,
        weight_decay = .1,
        momentum = .0,
        optimizer = optimizer.ADAM_W,
        scheduler = scheduler.ONE_CYCLE,
        kd_temperature = 20,
        kd_alpha = 0.5,
        drop_last = False,
    )

    # Adding additional info about training data
    BASE_DIR = Path("../..")
    PATH_DATA = 'data/feather-files'
    DATA_SET = BASE_DIR/PATH_DATA/config['data_set']

    with open(DATA_SET/'config.json', 'r') as fp:
        data_config = json.load(fp)
        config['window_size']       = int(data_config['maxw']) #maxw = max windowsize
        config['dimensions_out']    = int(data_config['maxl']) # maxl = max label length
        config['exclude_bacteria']  = ''.join(data_config['exclude']) # e = list of excluded bacteria

    ALPHABET_VAL      = list(config['alphabet'].values())
    ALPHABET_STR      = ''.join(ALPHABET_VAL)
    ALPHABET_SIZE     = len(ALPHABET_VAL)

    config['model_definition'] = "quartznet5x5.toml"
    config['model_name'], config['dimensions_out_scale'] = factory.get_model_details(config['model_definition'], config['window_size'])


    wandb.init(config=config)
    # THIS IS WHERE THE MAGIC HAPPENS
    config = wandb.config

    model = factory.bonito(config.window_size, config.device, config.model_definition, config.dropout)


    # Loss, metrics and callback
    _ctc_loss = metric.CtcLoss(config.dimensions_out_scale, config.batch_size, ALPHABET_SIZE)
    _kd_loss = metric.KdLoss(alpha=config.kd_alpha, temperature=config.kd_temperature, label_loss=_ctc_loss)
    loss = _kd_loss.loss() if config.knowledge_distillation else _ctc_loss.loss()

    metrics = [metric.ctc_accuracy(config.alphabet, 5)]


    # Load data
    if config.knowledge_distillation:
        data, teacher = f.load_training_data_with_teacher(DATA_SET, config.teacher_name)
        train_dl, valid_dl = prep.convert_to_dataloaders(data, split=.8, batch_size=config.batch_size, teacher=teacher, drop_last=config.drop_last)
    else:
        data = f.load_training_data(DATA_SET) 
        train_dl, valid_dl = prep.convert_to_dataloaders(data, split=.8, batch_size=config.batch_size, drop_last=config.drop_last)

    # Convert to databunch
    databunch = DataBunch(train_dl, valid_dl, device=config.device)

    # ## Model
    optimizer = optimizer.get_optimizer(config)
    scheduler = scheduler.get_scheduler(config)

    learner = Learner(databunch, model, loss_func=loss, metrics=metrics, opt_func=optimizer, callback_fns=WandbCallback).to_fp16()
    scheduler(learner)

    # Train

    learner.fit(config.epochs, lr=config.learning_rate, wd=config.weight_decay)


if __name__ == '__main__':
    main()
