#!/usr/bin/env python3
import jkbc.types as t
def get_bonito_config(config: t.Dict, double_kernel_sizes: bool = True):
    def convert_kernel_sizes(model_params):
        for k, v in model_params.items():
            if 'kernel' in k:
                model_params[k] = v*2+1
        return model_params

    if double_kernel_sizes:
      # Used for Wandb sweeps, which also outputs even integers.
      config = convert_kernel_sizes(config)
    
    model = {}
    
    blocks = 3 + config['b_blocks']
    model['block'] = [dict() for _ in range(blocks)]
    
    # Setting general values
    for i in range(blocks):
        block = model['block'][i]
        block['number'] = f'C{i}'
        block['repeat'] = 1
        block['stride'] = [1]
        block['dilation'] = [1]
        block['filters'] = None
        block['dropout'] = config['dropout']
        block['residual'] = False
        block['separable'] = False
        block['groups'] = 1
        block['shuffle'] = False
    
    model['labels'] =  dict()
    model['labels']['labels'] = ["N", "A", "C", "G", "T"]
    model['encoder'] = {'activation': 'relu'}
    model['input'] = {'features': 1}

    # C1
    model['block'][0]['stride'] = [config['c1_stride']]
    model['block'][0]['kernel'] = [config['c1_kernel']]
    model['block'][0]['filters'] = config['c1_filters']

    # B1 - B5
    for i in range(1, config['b_blocks'] + 1):
        model['block'][i]['number'] = f'B{i}'
        model['block'][i]['repeat'] = config[f'b{i}_repeat']
        model['block'][i]['filters'] = config[f'b{i}_filters']
        model['block'][i]['kernel'] = [config[f'b{i}_kernel']]
        model['block'][i]['dilation'] = [config[f'b{i}_dilation']]
        model['block'][i]['groups'] = config[f'b{i}_groups']
        model['block'][i]['residual'] = True
        model['block'][i]['separable'] = True

    # C2
    model['block'][-2]['kernel'] = [config['c2_kernel']]
    model['block'][-2]['filters'] = config['c2_filters']
    model['block'][-2]['separable'] = True

    # C3
    model['block'][-1]['kernel'] = [config['c3_kernel']]
    model['block'][-1]['filters'] = config['c3_filters']
    
    return model
