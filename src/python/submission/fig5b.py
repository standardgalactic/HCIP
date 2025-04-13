# Standard library imports
import importlib
from os.path import exists
import os.path
import sys
from pathlib import Path

# Third party imports
import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.stats as stats
import matplotlib

# Add parent directory to Python path for local imports
sys.path.append(str(Path(__file__).parents[1]))

# List of hyperparameter configurations to evaluate
hp_list = ['hp7', 'hp6', 'hp3', 'hp2', 'hp1', 'hp0']

# Constants for experiment setup
n_hp = len(hp_list)  # Number of hyperparameter configurations
n_candidates = 4  # Number of model candidates to compare
n_bootstrap = 50  # Number of bootstrap samples
config_wo_noise = ['hp7', 'hp6', 'hp3', 'hp2']  # Configs without noise injection

# Directory for saving model checkpoints and results
save_dir = Path(__file__).parents[1] / 'saved_data/fig4b/'

def _train_eval_model(i):
    """
    Train and evaluate a single model configuration.
    
    Args:
        i: Index of hyperparameter configuration to use
    """
    # Load hyperparameter configuration
    hp = hp_list[i]
    hyperparams = importlib.import_module('config.' + hp)
    meta_params = hyperparams.config

    # Create save directory if it doesn't exist
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Load and configure task parameters
    task_config = importlib.import_module(meta_params['task_config'])
    task_params = task_config.getConfig()

    # Set noise parameters based on configuration
    if hp in config_wo_noise:
        task_params['noise'] = 0.0
    else:
        task_params['wb_inc'] = 0.6

    task_params['rand_seed'] = i*5

    save_path = save_dir / hp
    print(save_path)

    # Initialize task data generator
    task_generator_module = importlib.import_module(meta_params['task_generator_module'])
    task_data_generator = getattr(task_generator_module, meta_params['task_class'])(task_params)

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
        # Load results for all models
        metric_all = np.zeros((n_hp, n_bootstrap, n_candidates))
        for i in range(n_hp):
            hp = hp_list[i]
            save_path = save_dir / hp
            results = torch.load(str(save_path) + '.nll', map_location=torch.device('cpu'))
            metric_all[i, :, :] = results['nll']

        # Set up plotting
        model_names = ['O','H', 'P', 'C']
        cmap = matplotlib.cm.get_cmap("cool", 4)
        fig, ax = plt.subplots()
        xticks = []
        xticklabels = []

        # Create violin plots for each hyperparameter configuration
        for i in range(n_hp):
            # Plot violin for each model's results
            violin = ax.violinplot(metric_all[i, :, :], positions=np.arange(n_candidates)+i*6, showmeans=False, showmedians=True)

            # Style violin plots
            for j in range(n_candidates):
                c = 'k'
                violin['bodies'][j].set_facecolor(c)
                violin['bodies'][j].set_edgecolor(c)
                violin['bodies'][j].set_alpha(0.3)
            c = 'r'
            
            # Style violin plot statistics
            namelist = ['cbars', 'cmins', 'cmaxes', 'cmedians']
            for name in namelist:
                violin[name].set_color([c for j in range(n_candidates)])
                violin[name].set_linewidth(1)

            # Update tick labels
            xticks+=(np.arange(n_candidates) + i*6).tolist()
            xticklabels+=model_names

            # Calculate and print statistical comparisons
            means = metric_all[i, :, :].mean(axis=0)
            argmin = np.argsort(means)
            best_model = metric_all[i, :, argmin[0]]
            second_best = metric_all[i, :, argmin[1]]
            
            # Calculate effect size and p-value
            from scipy import stats
            d = (np.mean(best_model) - np.mean(second_best)) / np.sqrt((np.var(best_model) + np.var(second_best)) / 2)
            t_stat, p_val = stats.ttest_rel(best_model, second_best)
            p_val_one_tailed = p_val / 2
            print(f"Effect size (Cohen's d) between {model_names[argmin[0]]} and {model_names[argmin[1]]}: {d:.2f}")
            print(f"One-tailed p-value: {p_val_one_tailed:.1e}")

        # Finalize plot formatting
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Model')
        ax.set_ylabel('Negative log likelihood')
        plt.show()
