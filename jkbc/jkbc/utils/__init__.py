def get_nested_dict(config, key):
    res = config[key]
    for k, v in dict(config).items():
        if '.' in k and k.split('.')[0] == key:
            inner_key = k.split('.')[1]
            res[inner_key] = v
    return res
