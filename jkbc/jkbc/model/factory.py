from jkbc.model.architectures.bonito import model as bonito


# +
def modify_config(identifier, config):
    if "TEST" == identifier:
        return test_modifier(config)
    elif "GROUPING" == identifier:
        return grouping(config)
    
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
    for con in _change_groups(config, [1,2,4,8], False):
        configs.append(con)
    for con in _change_groups(config, [2,4,8], True):
        configs.append(con)
        
    return configs
