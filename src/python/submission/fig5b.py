import importlib
from os.path import exists
import os.path

import matplotlib.pyplot as plt
import torch
import numpy as np
import scipy.stats as stats
import sys
from pathlib import Path
import matplotlib


# Add to Python path
sys.path.append(str(Path(__file__).parents[1]))

hp_list = ['hp7', 'hp6', 'hp3', 'hp2', 'hp1', 'hp0']

n_hp = len(hp_list)
n_candidates = 4
n_bootstrap = 50
config_wo_noise = ['hp7', 'hp6', 'hp3', 'hp2']
# read the checkpoint, which contains meta configuration
save_dir = Path(__file__).parents[1] / 'saved_data/fig4b/'

def _train_eval_model(i):
    hp = hp_list[i]
    hyperparams = importlib.import_module('config.' + hp)
    meta_params = hyperparams.config
    #

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # read task parameters
    task_config = importlib.import_module(meta_params['task_config'])
    task_params = task_config.getConfig()

    if hp in config_wo_noise:
        task_params['noise'] = 0.0
    else:
        task_params['wb_inc'] = 0.6

    task_params['rand_seed'] = i*5

    save_path = save_dir / hp
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
    if exists(str(save_path) + '.nll'):  # exists analysis result
        return
    
    if exists(str(save_path) + '.train'):  # exists a fully trained model
        saved_data = torch.load(str(save_path) + '.model', map_location=torch.device('cpu'))
        trained_net = saved_data['net']
    else:
        torch.random.seed()
        trained_net = trainer.train(training_params)

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

    #
    _ = analysis(params=analysis_params)

        

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # train on openmind
        job_id = int(sys.argv[1])
        site='openmind'
        _train_eval_model(int(job_id)-1)
    else:
        metric_all = np.zeros((n_hp, n_bootstrap, n_candidates))
        for i in range(n_hp):
            hp = hp_list[i]
            save_path = save_dir / hp
            results = torch.load(str(save_path) + '.nll', map_location=torch.device('cpu'))
                                
            metric_all[i, :, :] = results['nll']


        model_names = ['O','H', 'P', 'C']
        # violin plot
        cmap = matplotlib.cm.get_cmap("cool", 4)
        fig, ax = plt.subplots()
        xticks = []
        xticklabels = []
        for i in range(n_hp):
            violin = ax.violinplot(metric_all[i, :, :], positions=np.arange(n_candidates)+i*6, showmeans=False, showmedians=True)

            for j in range(n_candidates):
                c = 'k'
                violin['bodies'][j].set_facecolor(c)  # Body fill color
                violin['bodies'][j].set_edgecolor(c) # Body outline color
                violin['bodies'][j].set_alpha(0.3)
            c = 'r'
            namelist = ['cbars', 'cmins', 'cmaxes', 'cmedians']
            for name in namelist:
                violin[name].set_color([c for j in range(n_candidates)])
                violin[name].set_linewidth(1)

            
            xticks+=(np.arange(n_candidates) + i*6).tolist()
            xticklabels+=model_names

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel('Model')
        ax.set_ylabel('Negative log likelihood')
        plt.show()
        



