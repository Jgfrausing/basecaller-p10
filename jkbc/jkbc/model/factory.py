from jkbc.model.architectures.bonito import model as bonito


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
    return functions[identifier](config)

def test_modifier(config):
    ## Breaks run, because model will have too many parameters
    config['max_parameters'] = 2
    return [config]

def grouping(config):
    def _change_groups(config, groups, shuffle):
        for group in groups:
            con = dict(config)
            for block in [1,2,3,4,5]:
                con['model_params'][f'b{block}_groups'] = group
                con['model_params'][f'b{block}_shuffle'] = shuffle
            yield con
    
    configs = list(_change_groups(config, [1,2,4,8], False))
    configs += list(_change_groups(config, [2,4,8], True))
    
    return configs

def kernel_size(config):
    def _change_groups(config, kernel_scales):
        for scale in kernel_scales:
            con = dict(config)
            for block in [1,2,3,4,5]:
                con['model_params'][f'b{block}_kernel'] = int(scale*con['model_params'][f'b{block}_kernel'])
            yield con
            
    return list(_change_groups(config, [.95,.90,.85,.80,.75]))

def b_blocks(config):
    def _change_groups(config, groups):
        for g in groups:
            con = dict(config)
            con['model_params'][f'b_blocks'] = g
            yield con
            
    return list(_change_groups(config, [1,2,3,4]))

def filters(config):
    def _change_groups(config, filter_scales):
        for scale in filter_scales:
            con = dict(config)
            for block in [1,2,3,4,5]:
                con['model_params'][f'b{block}_filters'] = int(scale*con['model_params'][f'b{block}_filters'])
            yield con
            
    return list(_change_groups(config, [.95,.90,.85,.80,.75]))

def repeat(config):
    def _change_groups(config, repeats):
        for repeat in repeats:
            con = dict(config)
            for block in [1,2,3,4,5]:
                con['model_params'][f'b{block}_repeat'] = repeat
            yield con
    
    return list(_change_groups(config, [1,2,4,8]))

def dilation(config):
    def _change_groups(config, dilations):
        for dilation in dilations:
            con = dict(config)
            for block in [1,2,3,4,5]:
                con['model_params'][f'b{block}_dilation'] = dilation
            yield con
    
    return list(_change_groups(config, [2,3,4]))
