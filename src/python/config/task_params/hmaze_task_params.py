import numpy as np


#  numeric input, two arms
def getConfig():
    config = {
        'time_trial': 32,
        'time_pre_stim': 5,
        'grace_intv': 10,
        'velocity': 1,
        'length_arm1': np.arange(4, 8),
        'length_arm2': np.arange(4, 8),
        'length_arm3': np.arange(4, 8),
        'length_arm4': np.arange(4, 8),
        'length_arm5': np.arange(4, 8),
        'length_arm6': np.arange(4, 8),
        # 'time_cue': 17,
        'in_size': 8,  # 6 arms, 2 tms
        'out_size': 4,
        'D1': np.arange(0, 2),
        'D2': np.arange(0, 2),
        'ask_sample_size': 16384,
        'rep_per_maze': 10,
        'rand_seed': False,
        'wb': 0.15,
        'wb_inc': 0.6,
        'val_size': 0.3
    }

    return config


