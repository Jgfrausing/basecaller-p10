import jkbc.model.architectures.bonito as __bonito
#import jkbc.model.architectures.chiron as chiron

def bonito(device, base_dir): return __bonito.model(device, base_dir)
#def chiron(device, alphabet_size, bs, window_size=4096, dim_pred_out=4096//3): return chiron.model(window_size, dim_pred_out, device, alphabet_size, bs)
