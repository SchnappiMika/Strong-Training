import sys
from collections import OrderedDict
from os import path

import numpy as np
import torch
from yaml import dump

import cal_laica.utils.utils as utils
from cal_laica.environments.GridWorld import FourRoom, Cross, DistShift, GridWorld


class Config(object):
    def __init__(self, args):

        # SET UP PATHS
        self.paths = OrderedDict()
        self.paths['root'] = path.abspath(path.join(path.dirname(__file__), '..', '..'))

        # Do Hyper-parameter sweep, if needed
        self.idx = args.base + args.inc

        # Make results reproducible
        seed = args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Copy all the variables from args to config
        self.__dict__.update(vars(args))

        # Frequency of saving results and models.
        self.save_after = args.max_episodes // args.save_count if args.max_episodes > args.save_count else args.max_episodes

        # add path to models
        folder_suffix = args.experiment + args.folder_suffix
        self.paths['Experiments'] = path.join(self.paths['root'], 'outputs', 'Cal-LAICA')
        self.paths['experiment'] = path.join(self.paths['Experiments'], args.env_name, args.algo_name, folder_suffix)

        path_prefix = [self.paths['experiment']]
        self.paths['logs'] = path.join(*path_prefix, 'Logs/')
        self.paths['ckpt'] = path.join(*path_prefix, 'Checkpoints/')
        self.paths['results'] = path.join(*path_prefix, 'Results/')

        # Create directories
        for (key, val) in self.paths.items():
            if key not in ['root', 'datasets', 'data']:
                utils.create_directory_tree(val)

        # Save the all the configuration settings
        dump(args.__dict__, open(path.join(self.paths['experiment'], 'args.yaml'), 'w'), default_flow_style=False,
             explicit_start=True)

        # Output logging
        sys.stdout = utils.Logger(self.paths['logs'], args.restore, args.log_output)

        # Get the domain and algorithm
        self.env = self.get_domain(config=self, tag=args.env_tag, name=args.env_name)
        self.env.seed(seed)

        # Get the embedding paths
        self.get_embedding_paths(args)

        # Set Model
        # self.algo = utils.dynamic_load(path.join(self.paths['root'], 'Src', 'Algorithms'), args.algo_name,
        #                                load_class=True)

        self.feature_dim = [int(size) for size in args.NN_basis_dim.split(',')]
        self.policy_basis_dim = [int(size) for size in args.Policy_basis_dim.split(',')]

        # GPU
        self.device = torch.device('cpu')

        # optimizer
        if args.optim == 'adam':
            self.optim = torch.optim.Adam
        elif args.optim == 'rmsprop':
            self.optim = torch.optim.RMSprop
        elif args.optim == 'sgd':
            self.optim = torch.optim.SGD
        else:
            raise ValueError('Undefined type of optmizer')

        print("=====Configurations=====\n", args)

    def get_embedding_paths(self, args):
        if hasattr(args, 'true_embedding'):
            if self.env_name == 'Gridworld_CL':
                prefix = 'CL_' if self.fourier_order < 1 else 'CL_Fourier_'
                self.paths['embedding'] = utils.search(self.paths['root'],
                                                       prefix + str(args.change_count) + '_Grid' + str(args.n_actions),
                                                       exact=True)  # +'.pt'

            if (not self.gym_env) and self.true_embeddings:
                self.reduced_action_dim = self.env.get_embeddings().shape[1]

    def get_domain(self, config, tag, name):
        if tag in {"Cal_LAICA_GridWorld", "LAICA_GridWorld"}:
            if name == "four_room":
                env = FourRoom(n_actions=config.n_actions, max_episodes=config.max_episodes,
                               change_count=config.change_count)
            elif name == "crossing":
                env = Cross(n_actions=config.n_actions, max_episodes=config.max_episodes,
                            change_count=config.change_count)
            elif name == "Empty":
                env = GridWorld(
                    n_actions=config.n_actions, max_episodes=config.max_episodes,
                    change_count=config.change_count
                )
            elif name == "DistShift":
                env = DistShift(
                    n_actions=config.n_actions, max_episodes=config.max_episodes,
                    change_count=config.change_count
                )
            else:
                env = FourRoom(n_actions=config.n_actions, max_episodes=config.max_episodes,
                               change_count=config.change_count)
            return env
        else:
            return None
