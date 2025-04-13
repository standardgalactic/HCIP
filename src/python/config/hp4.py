"""
Attention: No
Objective function: Yes
Noise: Yes

"""

config = {
    'save_path': 'saved_data/hp4',

    'task_config': 'config.task_params.hmaze_task_params',

    'task_generator_module': 'task.HMaze',
    'task_class': 'hmaze_task',


    'net_config': 'config.net_params.rnn_params',


    'net_generator_module': 'nets.RNN',
    'net_class': 'RNN_att_NOISE',

    'trainer_config_module': 'config.train_params.rnn_training_params',
    'trainer_config_class': 'config_self_supervised',

    'trainer_module': 'trainer',
    'trainer_class': 'rnn_trainer',

    'analysis_module': 'analysis.analyses',
    'analysis_function': 'AnalysisMatchModelStat'
}
