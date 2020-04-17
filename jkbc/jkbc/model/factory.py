# +
import toml

from jkbc.model.architectures.bonito import model as bonito


# -

def get_model_details(definition, window_size):
    config = toml.load(definition)
    pred_out_scale = int(config['block'][0]['stride'][0])
    return definition, pred_out_scale
