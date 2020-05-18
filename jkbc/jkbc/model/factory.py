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
        'DILATION': dilation,
        'DILATION_KERNEL': dilation_and_kernel
    }
    con = copy.deepcopy(config)
    con['max_parameters'] = None
    con['knowledge_distillation'] = False
    configs, tags = functions[identifier](con, [identifier])
    
    return __remove_duplicates(configs), tags

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

def test_modifier(config, tags):
    ## Breaks run, because model will have too many parameters
    con = dict(config)
    config['max_parameters'] = 3
    con['max_parameters'] = 2
    return [config, config, con], tags

def grouping(config, tags):
    def _change_groups(config, groups, shuffle):
        for group in groups:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_groups'] = group
                con['model_params'][f'b{block}_shuffle'] = shuffle
            yield con
    
    configs = list(_change_groups(config, [4,2,8], False))
    configs += list(_change_groups(config, [2,4,8], True))
    
    return configs, tags

def kernel_size(config, tags):
    def _change_groups(config, kernel_scales):
        for scale in kernel_scales:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_kernel'] = int(scale*config['model_params'][f'b{block}_kernel'])
            yield con
        
    scales = list(np.arange(.5, 1, 0.05))
    return list(_change_groups(config, scales)), tags

def b_blocks(config, tags):
    def _change_groups(config, groups):
        for g in groups:
            con = copy.deepcopy(config)
            con['model_params'][f'b_blocks'] = g
            yield con
            
    return list(_change_groups(config, [1,2,3,4])), tags

def filters(config, tags):
    def _change_groups(config, filter_scales):
        for scale in filter_scales:
            if scale == 1:
                continue
            con = copy.deepcopy(config)
            
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_filters'] = int(scale*config['model_params'][f'b{block}_filters'])
            yield con
            
    scales = list(np.arange(.5, 1.55, 0.05))
    return list(_change_groups(config, scales)), tags

def repeat(config, tags):
    def _change_groups(config, repeats):
        for repeat in repeats:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_repeat'] = repeat
            yield con
    
    return list(_change_groups(config, [1,2,3,4,6,7,8,9])), tags

def dilation(config, tags):
    def _change_groups(config, dilations):
        for dilation in dilations:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_dilation'] = dilation
                con['model_params'][f'b{block}_kernel'] = _calculate_kernel_size(dilation, config['model_params'][f'b{block}_kernel'])
            yield con
    
    return list(_change_groups(config, [2,3,4])), tags

def dilation_and_kernel(config, tags):
    def _change_groups(config, dilations, kernel_scales):
        for dilation in dilations:
            for scale in kernel_scales:
                if 1 == dilation and 1 == scale:
                    #No change
                    continue
                    
                con = copy.deepcopy(config)
                for block in B_BLOCKS_LST:
                    con['model_params'][f'b{block}_dilation'] = dilation
                    kernel=int(config['model_params'][f'b{block}_kernel']*scale)
                    con['model_params'][f'b{block}_kernel'] = _calculate_kernel_size(dilation, kernel)
                yield con
            
    scales = list(np.arange(.5, 1.55, 0.05))
    return list(_change_groups(config, [2,3,4], scales)), ['KERNEL', 'DILATION']

def _calculate_kernel_size(dilation, old_kernel):
        '''
        (old-1)*(d-1)+old
        old: 1234       = 4
        d_2: 1*2*3*4    = 7  = 3*1+4
        d_3: 1**2**3**4 = 10 = 3*2+4
        '''
        
        return int((old_kernel-1)*(dilation-1)+old_kernel)
