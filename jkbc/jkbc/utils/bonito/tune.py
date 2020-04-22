#!/usr/bin/env python3
import jkbc.types as t
import uuid
import hashlib
def get_bonito_config(config: t.Dict):
    blocks = 3 + config['b_blocks']
    values_str = ''.join([str(val) for val in config.values()])
    hashed = hashlib.md5(values_str.encode()).hexdigest()
    model = dict(model = hashed, output_size = config['output_size'])
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
    model['encoder'] = {'activation': 'gelu'}
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


defult_model_config = dict(
    b_blocks = 5,
    dropout = 0.0,
    
    c1_stride = 3,
    c1_kernel = 33,
    c1_filters = 256,
    
    b1_repeat = 5,
    b1_filters = 256,
    b1_kernel = 33,
    b1_dilation = 1,

    b2_repeat = 5,
    b2_filters = 256,
    b2_kernel = 39,
    b2_dilation = 1,

    b3_repeat = 5,
    b3_filters = 512,
    b3_kernel = 51,
    b3_dilation = 1,

    b4_repeat = 5,
    b4_filters = 512,
    b4_kernel = 63,
    b4_dilation = 1,

    b5_repeat = 5,
    b5_filters = 512,
    b5_kernel = 75,
    b5_dilation = 1,

    c2_kernel = 87,
    c2_filters = 512,
    
    c3_kernel = 1,
    c3_filters = 1024,
)
