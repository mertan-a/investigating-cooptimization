import os
import shutil
from copy import deepcopy
import re

# filesystem related
def prepare_rundir(args):

    # decide where the experiment will be saved
    run_dir = "experiments/"
    dict_args = deepcopy(vars(args))

    # remove the arguments that are not relevant for the run_dir
    if 'gif_every' in dict_args:
        dict_args.pop('gif_every')

    # write the arguments
    for key_counter, key in enumerate(dict_args):
        # key can be None, skip it in that case
        if dict_args[key] is None:
            continue
        to_add = ''
        key_string = ''
        for k in key.split('_'):
            key_string += k[0]
        # if the key is a list, we need to iterate over the list
        if isinstance(dict_args[key], list) or isinstance(dict_args[key], tuple):
            to_add += '..' + key_string
            for i, item in enumerate(dict_args[key]):
                to_add += '-' + str(item)
        elif isinstance(dict_args[key], bool):
            if dict_args[key]:
                to_add += '..' + key_string + '-True'
        else:
            to_add = '..' + key_string + '-' + str(dict_args[key])

        if run_dir.endswith('/') and to_add.startswith('.'):
            run_dir += to_add[2:]
        else:
            run_dir += to_add
            
    return run_dir

def get_immediate_subdirectories_of(directory):
    # first check whether the directory exists
    if not os.path.exists(directory):
        return []
    # get all subdirectories
    return [name for name in os.listdir(directory)
            if os.path.isdir(os.path.join(directory, name))]

def get_files_in(directory, extension=None):
    # first check whether the directory exists
    if not os.path.exists(directory):
        return []
    # get all files
    if extension is None:
        return [name for name in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, name))]
    else:
        return [name for name in os.listdir(directory)
                if (os.path.isfile(os.path.join(directory, name)) and
                    os.path.splitext(name)[1] == extension)]

def create_folder_structure(rundir):
    if not os.path.exists(rundir):
        os.makedirs(rundir)
    if os.path.exists(os.path.join(rundir, 'pickled_population')):
        shutil.rmtree(os.path.join(rundir, 'pickled_population'))
    os.makedirs(os.path.join(rundir, 'pickled_population'))
    if os.path.exists(os.path.join(rundir, 'evolution_summary')):
        shutil.rmtree(os.path.join(rundir, 'evolution_summary'))
    os.makedirs(os.path.join(rundir, 'evolution_summary'))
    if os.path.exists(os.path.join(rundir, 'best_so_far')):
        shutil.rmtree(os.path.join(rundir, 'best_so_far'))
    os.makedirs(os.path.join(rundir, 'best_so_far'))
    if os.path.exists(os.path.join(rundir, 'to_record')):
        shutil.rmtree(os.path.join(rundir, 'to_record'))
    os.makedirs(os.path.join(rundir, 'to_record'))
    if os.path.exists(os.path.join(rundir, 'sub_simulations')):
        shutil.rmtree(os.path.join(rundir, 'sub_simulations'))
    os.makedirs(os.path.join(rundir, 'sub_simulations'))

# pareto front related
def natural_sort(l, reverse):
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key, reverse=reverse)


