import importlib
from os.path import exists
import os.path

import matplotlib.pyplot as plt
import torch
import numpy as np
import itertools
import scipy.stats as stats
from pathlib import Path
from multiprocessing import Pool

import sys
sys.path.append(str(Path(__file__).parents[1]))
import matplotlib
from scipy.ndimage import gaussian_filter1d

# read the checkpoint, which contains meta configuration
hyperparams = importlib.import_module('config.hp0')
meta_params = hyperparams.config

wb_inc_range = np.arange(0, 1.21, 0.1)

n_model_range = np.arange(5)  # 5 models per parameter
n_bootstrap = 10
n_candidates = 4
metric_all = np.zeros((len(wb_inc_range), len(n_model_range), n_bootstrap, n_candidates))
save_dir = Path(__file__).parents[1] / 'saved_data/fig5d/'

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

args = list(itertools.product(wb_inc_range, n_model_range))


def _train_eval_model(i):
    print(i, args)
    wb_inc, n_model = args[i]
    task_config = importlib.import_module(meta_params['task_config'])
    task_params = task_config.getConfig()

    task_params['wb_inc'] = wb_inc
    save_path = save_dir / (str(np.around(wb_inc, decimals=3)) + '_' + str(n_model))
    print(save_path)

    # load task generator and generator samples
    task_generator_module = importlib.import_module(meta_params['task_generator_module'])
    task_data_generator = getattr(task_generator_module, meta_params['task_class'])(task_params)
    task_data_generator.rand_seed = i

    # read NN parameters
    net_config = importlib.import_module(meta_params['net_config'])
    net_params = net_config.getConfig()
    net_params['task_data_generator'] = task_data_generator

    # load NN generator and initialize NN
    net_generator_module = importlib.import_module(meta_params['net_generator_module'])
    net = getattr(net_generator_module, meta_params['net_class'])(net_params)
    #######

    # load training parameters
    trainer_config_module = importlib.import_module(meta_params['trainer_config_module'])
    trainer_params = getattr(trainer_config_module, meta_params['trainer_config_class'])()

    # initialize trainer
    trainer_module = importlib.import_module(meta_params['trainer_module'])
    trainer_params['task_data_generator'] = task_data_generator
    trainer = getattr(trainer_module, meta_params['trainer_class'])(trainer_params)

    training_params = {
        'save_path': save_path,
        'net': net,
        'task_data_generator': task_data_generator
    }

    if exists(str(save_path) + '.nll'):  # exists analysis result
        return

    if exists(str(save_path) + '.train'):  # exists a fully trained model
        saved_data = torch.load(str(save_path) + '.model', map_location=torch.device('cpu'))
        trained_net = saved_data['net']
    else:
        torch.random.seed()
        trained_net = trainer.train(training_params)
    # trained_net = []

    # perform analysis
    analysis_module = importlib.import_module(meta_params['analysis_module'])
    analysis = getattr(analysis_module, 'AnalysisMatchModelStat')
    analysis_params = {
        'save_path': save_path,
        'net': trained_net,
        'trainer_params': trainer_params,
        'task_data_generator': task_data_generator,
        'n_bootstrap': n_bootstrap
    }


    _ = analysis(params=analysis_params)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # train on openmind
        job_id = int(sys.argv[1])
        site='openmind'
        _train_eval_model(int(job_id)-1)
    else:
        # plot results
        site='local'
        metric_all = np.zeros((len(wb_inc_range), len(n_model_range), n_bootstrap, n_candidates))

        for i in range(len(args)):
            wb_inc, n_model = args[i]
            save_path = save_dir / (str(np.around(wb_inc, decimals=3)) + '_' + str(n_model))
            results = torch.load(str(save_path) + '.nll', map_location=torch.device('cpu'))
            i_wb_inc = np.where(wb_inc_range == wb_inc)[0][0]
            i_model = np.where(n_model_range == n_model)[0][0]

            metric_all[i_wb_inc,i_model, :, :] = results['nll']

        # ttest between joint and counterfactual
        metric_all = metric_all.reshape(metric_all.shape[0], metric_all.shape[1]*metric_all.shape[2], metric_all.shape[3])
        metric_all = gaussian_filter1d(metric_all, sigma=1, axis=0)


        

        labels = ['joint','hierachical','postdictive', 'counterfactual']
        fig, ax = plt.subplots()
        cmap = matplotlib.cm.get_cmap("Set1", 9)
        for i in range(n_candidates):
            y = metric_all[:, :, i]
            x = wb_inc_range
            y_err = y.std(1) / np.sqrt(n_bootstrap)
            y_mean = y.mean(1)
            ax.plot(x, y_mean, color=cmap(i), label=labels[i])
            ax.fill_between(x, y_mean - y_err, y_mean + y_err, color=cmap(i), alpha=0.5)
        
        ax.set_xlabel('Counterfactual processing noise')
        ax.set_ylabel('Negative log likelihood')

        plt.legend()
        plt.show()


