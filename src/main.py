import os
import random
import numpy as np
import multiprocessing
import torch

from utils import prepare_rundir
from population import POPULATION
from algorithms import AFPO

import argparse
parser = argparse.ArgumentParser(description='run jobs')

# task 
parser.add_argument('--task', '-t', help='specify the task',
                    choices=['Walker-v0', 'BridgeWalker-v0'], default='Walker-v0')

# experiment related arguments
parser.add_argument('-jt', '--job_type', type=str, 
                    help='job type', choices=['optimizeBrain', 'cooptimize'])

# evolutionary algorithm related arguments
parser.add_argument('-nrp', '--nr_parents', type=int,
                     help='number of parents')
parser.add_argument('-nrg', '--nr_generations', type=int,
                     help='number of generations')
parser.add_argument('--nr_random_individual', '-nri', type=int, 
                    help='Number of random individuals to insert each generation')
# softrobot related arguments
parser.add_argument('--use_fixed_body', '-ufbo',
                    help='use fixed body/ies', action='store_true')
parser.add_argument('--fixed_bodies', '-fbo', nargs='+',
                    help='specify the fixed body/ies', type=str,  choices=['biped', 'worm', 'triped', 'block', 'deneme', 'centralized', 'decentralized', 'noobs'])
parser.add_argument('--fixed_body_path', '-fbp',
                    help='specify the path to the individual that contains the body you want', default=None)
parser.add_argument('--bounding_box', '-bb', nargs='+', type=int,
                    help='Bounding box dimensions (x,y). e.g.IND_SIZE=(6, 6)->workspace is a rectangle of 6x6 voxels') # trying to get rid of this
parser.add_argument('--use_pretrained_brain', '-uptbr',
                    help='use pretrained brain', action='store_true')
parser.add_argument('--pretrained_brain', '-ptbr',
                    help='specify the path to the pretrained brain\'s pkl')
parser.add_argument('--controller', '-ctrl', help='specify the controller',
                    choices=['DECENTRALIZED', 'CENTRALIZED', 'FIXED'], default='DECENTRALIZED')

parser.add_argument('-id', '--id', type=int, default=1,
                    help='id of the job')
parser.add_argument('-ge', '--gif_every', type=int, default=10,
                    help='generate gif every n generations')

args = parser.parse_args()

def run(args):

    multiprocessing.set_start_method('spawn')

    args.rundir = prepare_rundir(args)
    print('rundir', args.rundir)

    # if this experiment is currently running or has finished, we don't want to run it again
    if os.path.exists(args.rundir + '/RUNNING'):
        print('Experiment is already running')
        exit()
    if os.path.exists(args.rundir + '/FINISHED'):
        print('Experiment has already finished')
        exit()

    # Initializing the random number generator for reproducibility
    SEED = args.id
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # create population
    population = POPULATION(args=args)
    solution = population[0]

    # Setting up the optimization algorithm and runnning
    pareto_optimization = AFPO(args=args, population=population)
    pareto_optimization.optimize()

    # delete running file in any case
    if os.path.isfile(args.rundir + '/RUNNING'):
        os.remove(args.rundir + '/RUNNING')

    # write a file to indicate that the job finished successfully
    with open(args.rundir + '/FINISHED', 'w') as f:
        pass

if __name__ == '__main__':

    # sanity checks
    if args.job_type == 'cooptimize' and args.use_fixed_body == True:
        raise ValueError('cooptimization is not supported for fixed bodies')

    # run the job
    run(args)





