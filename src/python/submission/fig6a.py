# Standard library imports
import importlib
from os.path import exists
import torch
from pathlib import Path
import sys

# Add parent directory to Python path for local imports
sys.path.append(str(Path(__file__).parents[1]))

if __name__ == '__main__':
    # Load model configuration and hyperparameters from checkpoint
    checkpoint = importlib.import_module('config.hp0')
    meta_params = checkpoint.config
    
    # Set path to save model and analysis results
    save_path = Path(__file__).parents[1] / 'saved_data/fig4b/hp0'

    # Load task configuration parameters
    task_config = importlib.import_module(meta_params['task_config'])
    task_params = task_config.getConfig()

    # Initialize task data generator with configured parameters
    task_generator_module = importlib.import_module(meta_params['task_generator_module'])
    task_data_generator = getattr(task_generator_module, meta_params['task_class'])(task_params)

    # Load training configuration and parameters
    trainer_config_module = importlib.import_module(meta_params['trainer_config_module'])
    trainer_params = getattr(trainer_config_module, meta_params['trainer_config_class'])()

    # Load trained model from checkpoint if it exists
    if exists(str(save_path)+'.model'):  
        saved_data = torch.load(str(save_path)+'.model', map_location=torch.device('cpu'))
        trained_net = saved_data['net']
    else:
        raise Exception('No model checkpoint found at specified path!')

    # Run psychometric analysis on trained model
    analysis_module = importlib.import_module(meta_params['analysis_module'])
    analysis = getattr(analysis_module, 'Psychometric')
    
    # Configure analysis parameters
    analysis_params = {
        'net': trained_net,              # Trained neural network model
        'save_path': save_path,          # Path to save analysis results
        'trainer_params': trainer_params, # Training configuration
        'task_data_generator': task_data_generator  # Task data generator
    }
    
    # Execute analysis and store results
    results = analysis(params=analysis_params)
