# +
import argparse

from fastai.basics import *
import json
import wandb

import jkbc.model as m
import jkbc.model.factory as factory
import jkbc.constants as constants
import jkbc.files.fasta as fasta
import jkbc.files.torch_files as f
import jkbc.utils.kd as kd
import jkbc.utils.preprocessing as prep
import jkbc.utils.postprocessing as pop

BASE_DIR = '../..'

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--data_set", help="Path to preprossesed data folder", default=None)
parser.add_argument("-id", help="Identifier for run", default='3w9puesb')
parser.add_argument("-s", help="stride", default=300)
parser.add_argument("--save_teacher_output", help="Generate output for teacher", action='store_true')
parser.add_argument("--make_fasta_files", help="Generate output for teacher", action='store_true')
parser.add_argument("-bs", help="batch size",default=64)
parser.add_argument("-d", help="device",default='gpu')
parser.add_argument("--mapped_reads", help="Identifier for run", default=BASE_DIR/constants.MAPPED_READS)
parser.add_argument("--bacteria", help="Bacteria dictionary", default=BASE_DIR/constants.BACTERIA_DICT_PATH)

args = parser.parse_args()

# +
# Parameters
config = wandb.restore('config.yaml', run_path=f"kbargsteen/basecaller-p10-nbs_models/{args.id}")
with open(config.name, 'r') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

preprocessed_data = args.d
with open(preprocessed_data/'config.json', 'r') as fp:
    data_config = json.load(fp)
    window_size    = int(data_config['maxw']) #maxw = max windowsize
    dimensions_out = int(data_config['maxl']) # maxl = max label length
    min_label_len  = int(data_config['minl']) # minl = min label length

alphabet       = config['alphabet']
stride         = int(args.s)
device         = torch.device(args.d)

# +
# Model
model, _ = factory.bonito(config.window_size, device, model_config)
predicter = m.get_predicter(model, device).to_fp16()

weights = wandb.restore('bestmodel.pth', run_path=f"kbargsteen/basecaller-p10-nbs_models/{args.id}")
# fastai requires the name without .pth
model_weights = '.'.join(weights.name.split('.')[:-1])
predicter.load(model_weights)
model_weights


# +
def save_teacher_output(path, model_id, data, model, bs):
    # Read data from feather
    data = f.load_training_data(path)
    kd.generate_and_save_y_teacher(path, model_id, m.signal_to_input_tensor(data.x, DEVICE), model, bs=bs)

if args.save_teacher_output:
    save_teacher_output(preprocessed_data, args.id, data, predicter.model, bs)

# +
def get_accuracies(model_id, data_set, mapped_reads, bacteria_dict, labels_per_window, stride, window_size, alphabet, device, prediction_count):
    sc = prep.SignalCollection(mapped_reads, bacteria_dict, labels_per_window=labels_per_window, 
                               stride=stride, window_size=(window_size-1, window_size), blank_id=constants.BLANK_ID)
    
    alphabet_val   = list(alphabet.values())
    alphabet_str   = ''.join(alphabet_val)
    alphabet_size  = len(alphabet.keys())
    # predict range
    predict_objects = m.predict_range(predicter.model, sc, alphabet_str, window_size, 
                                      device, indexes=np.random.randint(0, len(sc), prediction_count))

    # convert outputs to dictionaries
    references = []
    predictions = []
    accuracies = []
    for po in predict_objects:
        [accuracies.append(a) for a in m.get_accuracies(po.references, po.predictions, alphabet_val)]
        ref, pred = fasta.map_decoded(po, alphabet_val, False)
        references.append(ref)
        predictions.append(pred)

    # make dicts ready to be saved
    ref_dict = fasta.merge(references)
    pred_dict = fasta.merge(predictions)

    # save fasta
    data_set_folder = data_set.split('/')[-1]
    fasta.save_dicts(pred_dict, ref_dict, f'{model_id}/predictions/{data_set_folder}')

if args.make_fasta_files:
    labels_per_window=(min_label_len,dimensions_out)
    get_accuracies(model_id, data_set, mapped_reads, bacteria_dict, 
                   labels_per_window, stride, window_size, alphabet, device, prediction_count)
