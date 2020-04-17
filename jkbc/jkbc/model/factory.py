# +
import toml

from jkbc.model.architectures.bonito import model as bonito


# -

def get_model_details(definition, window_size):
    config = toml.load(definition)
    model_name = config['name']
    pred_out_scale = int(config['block'][0]['stride'][0])
    return model_name, pred_out_scale
