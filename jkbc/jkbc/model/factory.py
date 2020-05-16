# +
import copy
import numpy as np

from jkbc.model.architectures.bonito import model as bonito
# -

B_BLOCKS_LST = [1,2,3,4,5]


# +
def modify_config(identifier, config):
    functions = {
        'TEST': test_modifier,
        'GROUPING': grouping,
        'KERNEL': kernel_size,
        'REPEAT': repeat,
        'FILTERS': filters,
        'BBLOCKS': b_blocks,
        'DILATION': dilation
    }
    configs = functions[identifier](config)
    
    return __remove_duplicates(configs)
def __remove_duplicates(configs):
    def string_config(d):
        h = ''
        for value in d.values():
            if type(value) is dict:
                h += string_config(value)
            else:
                h += f'#{value}'
        return h
    
    hashed_configs = {}
    for config in configs:
        hashed_configs[string_config(config)] = config
    return hashed_configs.values()

def test_modifier(config):
    ## Breaks run, because model will have too many parameters
    con = dict(config)
    config['max_parameters'] = 3
    con['max_parameters'] = 2
    return [config, config, con]

def grouping(config):
    def _change_groups(config, groups, shuffle):
        for group in groups:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_groups'] = group
                con['model_params'][f'b{block}_shuffle'] = shuffle
            yield con
    
    configs = list(_change_groups(config, [1,2,4,8], False))
    configs += list(_change_groups(config, [2,4,8], True))
    
    return configs

def kernel_size(config):
    def _change_groups(config, kernel_scales):
        for scale in kernel_scales:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_kernel'] = int(scale*config['model_params'][f'b{block}_kernel'])
            yield con
        
    scales = list(np.arange(.5, 1, 0.05))
    return list(_change_groups(config, scales))

def b_blocks(config):
    def _change_groups(config, groups):
        for g in groups:
            con = copy.deepcopy(config)
            con['model_params'][f'b_blocks'] = g
            yield con
            
    return list(_change_groups(config, [1,2,3,4]))

def filters(config):
    def _change_groups(config, filter_scales):
        for scale in filter_scales:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_filters'] = int(scale*config['model_params'][f'b{block}_filters'])
            yield con
            
    scales = list(np.arange(.5, 1, 0.05))
    return list(_change_groups(config, scales))

def repeat(config):
    def _change_groups(config, repeats):
        for repeat in repeats:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_repeat'] = repeat
            yield con
    
    return list(_change_groups(config, [1,2,4,8]))

def dilation(config):
    def _change_groups(config, dilations):
        for dilation in dilations:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_dilation'] = dilation
            yield con
    
    return list(_change_groups(config, [2,3,4]))
