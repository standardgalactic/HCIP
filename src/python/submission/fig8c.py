# Standard library imports
import importlib
from os.path import exists
import os.path

# Third party imports
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to Python path for local imports
sys.path.append(str(Path(__file__).parents[1]))

if __name__ == '__main__':
    # Load model configuration and hyperparameters from checkpoint
    checkpoint = importlib.import_module('config.hp0')
    checkpoint.config['analysis_function'] = 'SwitchFrequency'  # Set analysis type to switch frequency
    meta_params = checkpoint.config

    n_models = 10  # Number of models to analyze

    # Set path to save model and analysis results
    save_dir = Path(__file__).parents[1] / 'saved_data/fig5c/'

    # Create save directory if it doesn't exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Initialize array to store switch frequencies for each model
    switch_freq = np.zeros((n_models, 4))
    
    # Train and analyze each model
    for n in range(n_models):
        # Load task configuration parameters
        task_config = importlib.import_module(meta_params['task_config'])
        task_params = task_config.getConfig()

        save_path = save_dir / str(n)
        print(save_path)

        # Initialize task data generator with configured parameters
        task_generator_module = importlib.import_module(meta_params['task_generator_module'])
        task_data_generator = getattr(task_generator_module, meta_params['task_class'])(task_params)

        # Load and configure neural network parameters
        net_config = importlib.import_module(meta_params['net_config'])
        net_params = net_config.getConfig()
        net_params['task_data_generator'] = task_data_generator

        # Initialize neural network with configured parameters
        net_generator_module = importlib.import_module(meta_params['net_generator_module'])
        net = getattr(net_generator_module, meta_params['net_class'])(net_params)

        # Load and configure trainer parameters
        trainer_config_module = importlib.import_module(meta_params['trainer_config_module'])
        trainer_params = getattr(trainer_config_module, meta_params['trainer_config_class'])()

        # Initialize trainer with configured parameters
        trainer_module = importlib.import_module(meta_params['trainer_module'])
        trainer_params['task_data_generator'] = task_data_generator
        trainer = getattr(trainer_module, meta_params['trainer_class'])(trainer_params)

        # Configure training parameters
        training_params = {
            'save_path': save_path,
            'net': net,
            'task_data_generator': task_data_generator
        }

        # Load existing trained model or train new one
        if exists(str(save_path) + '.train'):
            saved_data = torch.load(str(save_path) + '.model', map_location=torch.device('cpu'))
            trained_net = saved_data['net']
        else:
            trained_net = trainer.train(training_params)

        # Configure analysis parameters
        analysis_module = importlib.import_module(meta_params['analysis_module'])
        analysis = getattr(analysis_module, meta_params['analysis_function'])
        analysis_params = {
            'save_path': save_path,
            'net': trained_net,
            'trainer_params': trainer_params,
            'task_data_generator': task_data_generator
        }

        # Load existing analysis results or run new analysis
        if exists(str(save_path) + '.sf'):
            results = torch.load(str(save_path) + '.sf', map_location=torch.device('cpu'))
        else:
            results = analysis(params=analysis_params)

        # Store switch frequency results
        switch_freq[n, :] = results['freq']

    # Plot results
    colormap = 'Set1'
    cmap = plt.get_cmap(colormap, 9)
    colors = cmap(np.arange(0, cmap.N))
    colors = np.flip(colors, axis=0)
    
    fig, ax = plt.subplots()
    switch_freq = np.flip(switch_freq, axis=1)
    handles = []
    
    # Create scatter plot with error bars for each horizontal difference
    for i in range(4):
        x = np.ones(n_models)*i
        y = switch_freq[:, i]
        y_mean = np.mean(y)
        y_err = np.percentile(y, 95)-y_mean
        ax.scatter(x, y, color='k')
        ax.errorbar(i+0.15, y_mean, yerr=y_err, fmt='o', capsize=5, color=(.8,.2,.1))

    # Set axis labels
    ax.set_xlabel('Horizontal Diff.')
    ax.set_ylabel('Mean switches')
    plt.show()
