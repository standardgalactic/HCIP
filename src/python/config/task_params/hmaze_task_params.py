import numpy as np


#  numeric input, two arms
def getConfig():
    config = {
        'time_trial': 32,        # Total duration of each trial
        'time_pre_stim': 5,      # Time before stimulus presentation
        'grace_intv': 10,        # Grace interval for response
        'velocity': 1,           # Movement velocity in the maze
        'length_arm1': np.arange(4, 8),  # Possible lengths for arm 1
        'length_arm2': np.arange(4, 8),  # Possible lengths for arm 2  
        'length_arm3': np.arange(4, 8),  # Possible lengths for arm 3
        'length_arm4': np.arange(4, 8),  # Possible lengths for arm 4
        'length_arm5': np.arange(4, 8),  # Possible lengths for arm 5
        'length_arm6': np.arange(4, 8),  # Possible lengths for arm 6
        # 'time_cue': 17,        # Cue presentation time (commented out)
        'in_size': 8,            # Input size: 6 arms + 2 timing signals
        'out_size': 4,           # Output size (number of possible actions)
        'D1': np.arange(0, 2),   # Decision point 1 options (0 or 1)
        'D2': np.arange(0, 2),   # Decision point 2 options (0 or 1)
        'ask_sample_size': 16384,  # Number of samples to generate
        'rep_per_maze': 10,      # Number of repetitions per maze configuration
        'rand_seed': False,      # Whether to use random seed
        'wb': 0.15,              # Base width parameter
        'wb_inc': 0.6,           # Width increment parameter
        'val_size': 0.3          # Validation set proportion
    }

    return config


