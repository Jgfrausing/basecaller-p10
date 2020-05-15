from jkbc.model.architectures.bonito import model as bonito


# +
def modify_config(identifier, config):
    if "TEST" == identifier:
        return test_modifier(config)
    elif "GROUPING" == identifier:
        return grouping(config)
    elif "KERNEL" == identifier:
        return kernel_size(config)
    
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
