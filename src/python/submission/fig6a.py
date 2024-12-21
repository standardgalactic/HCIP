import importlib
from os.path import exists
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parents[1]))

if __name__ == '__main__':
    # read the checkpoint, which contains meta configuration
    checkpoint = importlib.import_module('config.hp0')
    meta_params = checkpoint.config
    save_path = Path(__file__).parents[1] / 'saved_data/fig4b/hp0'

    # read task parameters
    task_config = importlib.import_module(meta_params['task_config'])
    task_params = task_config.getConfig()

    # load task generator and generator samples
    task_generator_module = importlib.import_module(meta_params['task_generator_module'])
    task_data_generator = getattr(task_generator_module, meta_params['task_class'])(task_params)

    # load training parameters
    trainer_config_module = importlib.import_module(meta_params['trainer_config_module'])
    trainer_params = getattr(trainer_config_module, meta_params['trainer_config_class'])()


    


    if exists(str(save_path)+'.model'):  # exists a fully trained model
        saved_data = torch.load(str(save_path)+'.model', map_location=torch.device('cpu'))
        trained_net = saved_data['net']
    else:
        raise Exception('No model!')

    # perform analysis
    analysis_module = importlib.import_module(meta_params['analysis_module'])
    analysis = getattr(analysis_module, 'Psychometric')
    analysis_params = {
        'net': trained_net,
        'save_path': save_path,
        'trainer_params': trainer_params,
        'task_data_generator': task_data_generator
    }
    results = analysis(params=analysis_params)





