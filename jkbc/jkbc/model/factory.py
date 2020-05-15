from jkbc.model.architectures.bonito import model as bonito


# +
def modify_config(identifier, config):
    if "TEST" == identifier:
        return test_modifier(config)
    
def test_modifier(config):
    config['max_parameters'] = 2
    return [config]
        
