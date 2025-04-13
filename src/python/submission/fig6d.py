# Standard library imports
import importlib
from os.path import exists
import os.path

# Third party imports
import matplotlib.pyplot as plt
import torch
import numpy as np
import itertools
import scipy.stats as stats
from pathlib import Path
from multiprocessing import Pool

# Add parent directory to Python path for local imports
import sys
sys.path.append(str(Path(__file__).parents[1]))
import matplotlib
from scipy.ndimage import gaussian_filter1d

# Load model configuration and hyperparameters from checkpoint
hyperparams = importlib.import_module('config.hp0')
meta_params = hyperparams.config

# Define range of counterfactual processing noise values to test
wb_inc_range = np.arange(0, 1.21, 0.1)

# Experiment parameters
n_model_range = np.arange(5)  # 5 models per parameter
n_bootstrap = 10              # Number of bootstrap samples
n_candidates = 4             # Number of model candidates to compare

# Initialize array to store metrics for all models and conditions
metric_all = np.zeros((len(wb_inc_range), len(n_model_range), n_bootstrap, n_candidates))

# Set up directory for saving results
save_dir = Path(__file__).parents[1] / 'saved_data/fig5d/'

# Create save directory if it doesn't exist
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)

# Generate all combinations of noise values and model indices
args = list(itertools.product(wb_inc_range, n_model_range))


def _train_eval_model(i):
    """
    Train and evaluate a single model with specific noise level.
    
    Args:
        i: Index into args list specifying which model/noise combination to run
    """
    print(i, args)
    wb_inc, n_model = args[i]
    
    # Load and configure task parameters
    task_config = importlib.import_module(meta_params['task_config'])
    task_params = task_config.getConfig()

    # Set noise level for this run
    task_params['wb_inc'] = wb_inc
    save_path = save_dir / (str(np.around(wb_inc, decimals=3)) + '_' + str(n_model))
    print(save_path)

    # Initialize task data generator
    task_generator_module = importlib.import_module(meta_params['task_generator_module'])
    task_data_generator = getattr(task_generator_module, meta_params['task_class'])(task_params)
    task_data_generator.rand_seed = i

    # Load and configure neural network parameters
    net_config = importlib.import_module(meta_params['net_config'])
    net_params = net_config.getConfig()
    net_params['task_data_generator'] = task_data_generator

    # Initialize neural network
    net_generator_module = importlib.import_module(meta_params['net_generator_module'])
    net = getattr(net_generator_module, meta_params['net_class'])(net_params)

    # Load and configure trainer parameters
    trainer_config_module = importlib.import_module(meta_params['trainer_config_module'])
    trainer_params = getattr(trainer_config_module, meta_params['trainer_config_class'])()

    # Initialize trainer
    trainer_module = importlib.import_module(meta_params['trainer_module'])
    trainer_params['task_data_generator'] = task_data_generator
    trainer = getattr(trainer_module, meta_params['trainer_class'])(trainer_params)

    training_params = {
        'save_path': save_path,
        'net': net,
        'task_data_generator': task_data_generator
    }

    # Skip if analysis results already exist
    if exists(str(save_path) + '.nll'):
        return

    # Load existing trained model or train new one
    if exists(str(save_path) + '.train'):
        saved_data = torch.load(str(save_path) + '.model', map_location=torch.device('cpu'))
        trained_net = saved_data['net']
    else:
        torch.random.seed()
        trained_net = trainer.train(training_params)

    # Run analysis on trained model
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
        # Run single model training job (for cluster)
        job_id = int(sys.argv[1])
        site='openmind'
        _train_eval_model(int(job_id)-1)
    else:
        # Run full analysis and plotting
        site='local'
        metric_all = np.zeros((len(wb_inc_range), len(n_model_range), n_bootstrap, n_candidates))

        # Load results for all models
        for i in range(len(args)):
            wb_inc, n_model = args[i]
            save_path = save_dir / (str(np.around(wb_inc, decimals=3)) + '_' + str(n_model))
            results = torch.load(str(save_path) + '.nll', map_location=torch.device('cpu'))
            i_wb_inc = np.where(wb_inc_range == wb_inc)[0][0]
            i_model = np.where(n_model_range == n_model)[0][0]

            metric_all[i_wb_inc,i_model, :, :] = results['nll']

        # Reshape and smooth results
        metric_all = metric_all.reshape(metric_all.shape[0], metric_all.shape[1]*metric_all.shape[2], metric_all.shape[3])
        metric_all = gaussian_filter1d(metric_all, sigma=1, axis=0)

        # Plot results
        labels = ['joint','hierachical','postdictive', 'counterfactual']
        fig, ax = plt.subplots()
        cmap = matplotlib.cm.get_cmap("Set1", 9)
        
        # Create line plot for each model type
        for i in range(n_candidates):
            y = metric_all[:, :, i]
            x = wb_inc_range
            y_err = y.std(1) / np.sqrt(n_bootstrap)  # Standard error
            y_mean = y.mean(1)
            ax.plot(x, y_mean, color=cmap(i), label=labels[i])
            ax.fill_between(x, y_mean - y_err, y_mean + y_err, color=cmap(i), alpha=0.5)
        
        # Set axis labels and show plot
        ax.set_xlabel('Counterfactual processing noise')
        ax.set_ylabel('Negative log likelihood')

        plt.legend()
        plt.show()

