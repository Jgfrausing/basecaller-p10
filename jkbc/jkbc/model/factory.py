# +
import copy
import numpy as np

from jkbc.model.architectures.bonito import model as bonito
# -

B_BLOCKS_LST = [1,2,3,4,5]
SCALES = list(np.arange(.7, 1.4, 0.1))


# +
def modify_config(identifier, config):
    functions = {
        'TEST': test_modifier,
        'BONITO': identity,
        'SMALL': lambda c, t: (_change_filters(config, [0.5]), ['SMALL', 'FILTERS']),
        'LARGE': lambda c, t: (_change_filters(config, [1.5]), ['LARGE', 'FILTERS']),
        'GROUPING': grouping,
        'REPEAT': repeat,
        'FILTERS': lambda c, t: (_change_filters(config, SCALES), t),
        'BBLOCKS': b_blocks,
        'DILATION_KERNEL': lambda c, t: (_change_dilation_and_kernel(config, [1,2,3], SCALES), t),
        'DILATION_1_KERNEL': lambda c, t: (_change_dilation_and_kernel(config, [1], SCALES), t),
        'DILATION_2_KERNEL': lambda c, t: (_change_dilation_and_kernel(config, [2], SCALES), t),
        'DILATION_3_KERNEL': lambda c, t: (_change_dilation_and_kernel(config, [3], SCALES), t),
        'REPEAT_EXTRA_6_7_8': lambda c, t: repeat_extra(c, t, [6, 7, 8]),
        'REPEAT_EXTRA_9_10_11': lambda c, t: repeat_extra(c, t, [9, 10, 11]),
    }
    configs, tags = functions[identifier](config, [identifier])
    
    return _remove_duplicates(configs), tags

def random_permutation(function_identifiers, config):
    modified_fn = lambda fid, con: list(modify_config(fid, con)[0])
    flatten = lambda l: list([item for sublist in l for item in sublist])
    configs = [config]
    
    # Getting all permutations
    for fn_id in function_identifiers:
        configs = flatten([modified_fn(fn_id, c) for c in configs]) + configs
        configs = list(_remove_duplicates(configs))
    config_len = len(configs)
    
    # Getting all configs run without combinations
    executed_configs = [config]
    for fn_id in function_identifiers:
        tmp = modified_fn(fn_id, config)
        executed_configs += tmp
    len_executed_configs = len(executed_configs)
    
    # Remove unwanted configs
    configs_ = _remove_configs(configs,executed_configs)
    new_config_len =  len(configs_)
    assert config_len-new_config_len == len_executed_configs, f"Something went wrong: {config_len-new_config_len} != {new_config_len}"

    return configs_

def _string_config(d):
        h = ''
        for key, value in d.items():
            if type(value) is dict:
                h += _string_config(value)
            else:
                h += f'##{key}#{value}'
        return h

def _hash_configs(configs):
    hashed_configs = {}
    for config in configs:
        hashed_configs[_string_config(config)] = config
    return hashed_configs
    
def _remove_configs(a, b):
    def remove(a,b):
        return [v for k, v in a.items() if k not in b.keys()]
    
    hashed_a = _hash_configs(a)
    hashed_b = _hash_configs(b)
    
    return remove(hashed_a, hashed_b)

def _remove_duplicates(configs):
    def string_config(d):
        h = ''
        for value in d.values():
            if type(value) is dict:
                h += string_config(value)
            else:
                h += f'#{value}'
        return h
    
    hashed_configs = _hash_configs(configs)
    return hashed_configs.values()

def identity(config, tags):
    return [config], tags

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
    
    configs = list(_change_groups(config, [2,4,8], False))
    configs += list(_change_groups(config, [2,4,8], True))
    
    return configs, tags

def b_blocks(config, tags):
    def _change_config(config, blocks):
        for b in blocks:
            con = copy.deepcopy(config)
            con['model_params'][f'b_blocks'] = b
            for block in B_BLOCKS_LST:
                if block > b:
                    con['model_params'][f'b{block}_dilation'] = None
                    con['model_params'][f'b{block}_filters'] = None
                    con['model_params'][f'b{block}_kernel'] = None
                    con['model_params'][f'b{block}_repeat'] = None
                    con['model_params'][f'b{block}_groups'] = None
                    con['model_params'][f'b{block}_shuffle'] = None
            yield con
            
    return list(_change_config(config, [1,2,3,4])), tags

def _change_filters(config, filter_scales):
    def change(config, filter_scales):
        for scale in filter_scales:
            if scale == 1:
                continue
            con = copy.deepcopy(config)

            for block in B_BLOCKS_LST:
                filter_ = int(scale*config['model_params'][f'b{block}_filters'])
                # We convert the filter st it is divisible with 8 in order to accommodate grouping of sizes up to 8
                con['model_params'][f'b{block}_filters'] = filter_ - filter_ % 8
            yield con
    return list(change(config, filter_scales))

def repeat(config, tags):
    def _change_groups(config, repeats):
        for repeat in repeats:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_repeat'] = repeat
            yield con
    
    return list(_change_groups(config, [1,2,3,4,6,7,8,9])), tags

def _change_dilation_and_kernel(config, dilations, kernel_scales):
    def change(config, dilations, kernel_scales):
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
    return list(change(config, dilations, kernel_scales))
                
def _calculate_kernel_size(dilation, simulated_kernel_size):
        '''
        args:
            dilation: dilation size
            simulated_kernel_size: requested span of kernel when dilation is used
        '''
        
        '''
        (kernel_size-1)*(dilation-1)+no_dilation = simulated_kernel_size
        kernel_size = (simulated_kernel_size+dilation-1)/dilation
                   wanted ks    sim ks               calculated ks  
        1: 12345 = 5 => 1*0+5 = 5 => (4 + 1 - 1)/1 = 4
        2: 1_2_3 = 3 => 2*1+3 = 5 => (4 + 2 - 1)/2 = 2.5
        3: 1__2  = 2 => 1*2+2 = 4 => (4 + 3 - 1)/3 = 2
        '''
        return int((simulated_kernel_size + dilation - 1)/dilation)

      
      
def repeat_extra(config, tags, values):
    print(f"Starting runs with the following extra repeats: {values}.")
    def _change_groups(config, repeats):
        for repeat in repeats:
            con = copy.deepcopy(config)
            for block in B_BLOCKS_LST:
                con['model_params'][f'b{block}_repeat'] = repeat
            yield con
    
    return list(_change_groups(config, values)), tags