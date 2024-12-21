import importlib
from os.path import exists

import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))
# read the checkpoint, which contains meta configuration

if __name__ == '__main__':
    checkpoint = importlib.import_module('config.hp0')
    checkpoint.config['analysis_function'] = 'saccade'
    meta_params = checkpoint.config

    save_path = Path(__file__).parents[1] / 'saved_data/fig4b/hp0'

    # read task parameters
    task_config = importlib.import_module(meta_params['task_config'])
    task_params = task_config.getConfig()



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

    results = analysis(params=analysis_params)



