# +
import yaml
import json
import hashlib
from collections import OrderedDict

from fastai.basics import *
import wandb
from wandb.wandb_config import Config
from wandb.fastai import WandbCallback
# -

import jkbc.model as m
import jkbc.model.factory as factory
import jkbc.model.optimizer as optim
import jkbc.model.scheduler as sched
import jkbc.files.torch_files as f
import jkbc.model.metrics as metric
import jkbc.utils.preprocessing as prep
import jkbc.utils.bonito.tune as bonito

# +
def get_nested_dict(config, key):
    res = config[key]
    for k, v in dict(config).items():
        if '.' in k and k.split('.')[0] == key:
            inner_key = k.split('.')[1]
            res[inner_key] = v
    return res

def hash_config(config):
    values = OrderedDict(dict(config)).values()
    encoded = hashlib.md5(''.join(str(values)).encode())
    return encoded.hexdigest()


# +
DEVICE = torch.device('cuda')# m.get_available_gpu() 

with open('config_default.yaml', 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
config['device'] = DEVICE

# Use to override dataset from wandb config
data_set = config['data_set'] if config['use_default_data_set'] else None

wandb.init(config=config) #, resume='allow', id='bonito_stride6')
wandb.run.save()
# THIS IS WHERE THE MAGIC HAPPENS
# fix to issue https://github.com/wandb/client/issues/982
config = wandb.config
config.id = wandb.run.id
if data_set:
    config.data_set = data_set
print(f'Resumed: {wandb.run.resumed}\nId: {config.id}\nDevice: {config.device}')

# +
# Setting constants using config
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

# +
# get model
model_params = get_nested_dict(config, 'model_params')
model_config = bonito.get_bonito_config(config.model_params)
model_config['output_size'] = config['output_size']


model, config.dimensions_out_scale = factory.bonito(config.window_size, config.device, model_config)
    
config.parameters = m.get_parameter_count(model)

if config.max_paramters < config.parameters:
    raise ValueError("Too many parameters ({paramaters})")

# +
# Loss, metrics and callback
_ctc_loss = metric.CtcLoss(config.dimensions_out_scale, config.batch_size, ALPHABET_SIZE)
_kd_loss = metric.KdLoss(alpha=config.kd_alpha, temperature=config.kd_temperature, label_loss=_ctc_loss)
loss = _kd_loss.loss() if config.knowledge_distillation else _ctc_loss.loss()

metrics = [metric.ctc_accuracy(config.alphabet, 5)]

# +
# Load data
if config.knowledge_distillation:
    data, teacher = f.load_training_data_with_teacher(DATA_SET, config.teacher_name)
    train_dl, valid_dl = prep.convert_to_dataloaders(data, split=.8, batch_size=config.batch_size, teacher=teacher, drop_last=config.drop_last)
else:
    data = f.load_training_data(DATA_SET) 
    train_dl, valid_dl = prep.convert_to_dataloaders(data, split=.8, batch_size=config.batch_size, drop_last=config.drop_last)

# Convert to databunch
databunch = DataBunch(train_dl, valid_dl, device=config.device)

# +
# Learner
optimizer = optim.get_optimizer(config)
scheduler = sched.get_scheduler(config)

learner = Learner(databunch, model, loss_func=loss, metrics=metrics, opt_func=optimizer, callback_fns=WandbCallback).to_fp16()
scheduler(learner)

if wandb.run.resumed:
    try:
        weights = wandb.restore('bestmodel.pth')
        # fastai requirest the name without .pth
        weight_name = '.'.join(weights.name.split('.')[:-1])
        learner.load(weight_name)
    except:
        print('No model found')
# -

# Train
learner.fit(config.epochs, lr=config.learning_rate, wd=config.weight_decay)


