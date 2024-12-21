import importlib
from os.path import exists
import os.path

import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
# read the checkpoint, which contains meta configuration

if __name__ == '__main__':
    checkpoint = importlib.import_module('config.hp0')
    checkpoint.config['analysis_function'] = 'SwitchFrequency'
    meta_params = checkpoint.config

    n_models = 10  # 5 models per parameter

    save_dir = Path(__file__).parents[1] / 'saved_data/fig5c/'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    switch_freq = np.zeros((n_models, 4))
    for n in range(n_models):
        # read task parameters
        task_config = importlib.import_module(meta_params['task_config'])
        task_params = task_config.getConfig()

        save_path = save_dir / str(n)
        print(save_path)

        # load task generator and generator samples
        task_generator_module = importlib.import_module(meta_params['task_generator_module'])
        task_data_generator = getattr(task_generator_module, meta_params['task_class'])(task_params)

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

        if exists(str(save_path) + '.train'):  # exists a fully trained model
            saved_data = torch.load(str(save_path) + '.model', map_location=torch.device('cpu'))
            trained_net = saved_data['net']
        else:
            trained_net = trainer.train(training_params)

        # perform analysis
        analysis_module = importlib.import_module(meta_params['analysis_module'])
        analysis = getattr(analysis_module, meta_params['analysis_function'])
        analysis_params = {
            'save_path': save_path,
            'net': trained_net,
            'trainer_params': trainer_params,
            'task_data_generator': task_data_generator
        }

        if exists(str(save_path) + '.sf'):  # exists a fully trained model
            results = torch.load(str(save_path) + '.sf', map_location=torch.device('cpu'))
        else:
            results = analysis(params=analysis_params)

        switch_freq[n, :] = results['freq']


    colormap = 'Set1'
    cmap = plt.get_cmap(colormap, 9)
    colors = cmap(np.arange(0, cmap.N))
    colors = np.flip(colors, axis=0)
    fig, ax = plt.subplots()
    switch_freq = np.flip(switch_freq, axis=1)
    handles = []
    for i in range(4):
        x = np.ones(n_models)*i
        y = switch_freq[:, i]
        y_mean = np.mean(y)
        y_err = np.percentile(y, 95)-y_mean
        ax.scatter(x, y, color='k')
        ax.errorbar(i+0.15, y_mean, yerr=y_err, fmt='o', capsize=5, color=(.8,.2,.1))

    ax.set_xlabel('Horizontal Diff.')
    ax.set_ylabel('Mean switches')
    plt.show()
