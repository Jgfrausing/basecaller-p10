# +
import os

import jkbc.types as t
import jkbc.utils.postprocessing as pop


# -

def get_stats(prediction: t.Tensor2D, actual: str, alphabet: t.List[str], beam_sizes: t.List[int]):
    print(actual)
    y_pred_index = prediction[None,:,:]
    for beam in beam_sizes:
        decoded = pop.decode(y_pred_index, threshold=.0, beam_size=beam, alphabet=alphabet)   
        predicted = decoded[0]
        accuracy = pop.calc_accuracy(actual, predicted)
        yield (predicted, beam, accuracy)


def get_notebook_name():
    from notebook import notebookapp
    import urllib
    import json
    import ipykernel

    """Returns the absolute path of the Notebook or None if it cannot be determined
    NOTE: works only when the security is token-based or there is also no password
    """
    connection_file = os.path.basename(ipykernel.get_connection_file())
    kernel_id = connection_file.split('-', 1)[1].split('.')[0]

    for srv in notebookapp.list_running_servers():
        try:
            if srv['token']=='' and not srv['password']:  # No token and no password, ahem...
                req = urllib.request.urlopen(srv['url']+'api/sessions')
            else:
                req = urllib.request.urlopen(srv['url']+'api/sessions?token='+srv['token'])
            # Get correct session
            sessions = json.load(req)
            
            for sess in [s for s in sessions if s['kernel']['id'] == kernel_id]:
                # Get path and filename
                nb_path = os.path.join(srv['notebook_dir'],sess['notebook']['path'])
    
            return __get_file_name(nb_path)
        except:
            pass  # There may be stale entries in the runtime directory 
    raise Exception("Failed to get the path of the notebook")


def get_newest_model(folder_path: t.PathLike):
    import glob
    
    list_of_files = glob.glob(os.path.join(folder_path, '*')) # * means all if need specific format then *.csv
    return __get_file_name(max(list_of_files, key=os.path.getctime))


def __get_file_name(file_path: t.PathLike):
    file_name, _ = os.path.splitext(os.path.basename(file_path))
    return file_name
    
