import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

# This class takes parameters of the task_params, and generates the teacher input/output of the task_params
# The task simulates navigation through an H-shaped maze with variable arm lengths and noise

class hmaze_task(object):
    def __init__(self, params):
        # Initialize task parameters from input dictionary
        self.time_trial = params['time_trial']  # Total time steps per trial
        self.time_pre_stim = params['time_pre_stim']  # Time before stimulus onset
        self.length_arm1 = params['length_arm1']  # Length of first arm
        self.length_arm2 = params['length_arm2']  # Length of second arm  
        self.length_arm3 = params['length_arm3']  # Length of third arm
        self.length_arm4 = params['length_arm4']  # Length of fourth arm
        self.length_arm5 = params['length_arm5']  # Length of fifth arm
        self.length_arm6 = params['length_arm6']  # Length of sixth arm
        self.velocity = params['velocity']  # Movement velocity
        self.D1 = params['D1']  # First decision point
        self.D2 = params['D2']  # Second decision point
        self.ask_sample_size = params['ask_sample_size']  # Number of samples to generate
        self.in_size = params['in_size']  # Input dimension
        self.out_size = params['out_size']  # Output dimension
        self.rep_per_maze = params['rep_per_maze']  # Repetitions per maze configuration
        self.val_size = params['val_size']  # Validation set size
        self.rand_seed = params['rand_seed']  # Random seed
        self.wb = params['wb']  # Weber fraction (noise level)
        self.wb_inc = params['wb_inc']  # Weber fraction increment

    def __call__(self, seed=None):
        # Extract parameters for convenience
        time_trial = self.time_trial
        l1, l2, l3 = self.length_arm1, self.length_arm2, self.length_arm3
        l4, l5, l6 = self.length_arm4, self.length_arm5, self.length_arm6
        d1, d2 = self.D1, self.D2
        ask_sample_size = self.ask_sample_size
        in_size = self.in_size
        out_size = self.out_size
        rep = self.rep_per_maze
        time_pre_stim = self.time_pre_stim
        wb = self.wb

        # Calculate total number of conditions and trials
        n_conditions = len(l1) * len(l2) * len(l3) * len(l4) * len(l5) * len(l6) * len(d1) * len(d2)
        n_total = n_conditions * rep

        # Generate meshgrid for all input combinations
        mesh = np.meshgrid(l1, l2, l3, l4, l5, l6, d1, d2, np.arange(rep))
        l1_list = mesh[0].flatten()
        l2_list = mesh[1].flatten()
        l3_list = mesh[2].flatten()
        l4_list = mesh[3].flatten()
        l5_list = mesh[4].flatten()
        l6_list = mesh[5].flatten()
        d1_list = mesh[6].flatten()
        d2_list = mesh[7].flatten()

        # Initialize arrays for inputs and labels
        in_arms = np.zeros((n_total, in_size - 2, time_trial))  # Arm lengths
        in_time = np.zeros((n_total, 2, time_trial))  # Time information
        labels = np.zeros((n_total, out_size, time_trial))  # Output labels

        # Generate data for each trial
        for i in range(n_total):
            # Set arm lengths for current trial
            in_arms[i, 0:6, :] = [l1_list[i], l2_list[i], l3_list[i], 
                                 l4_list[i], l5_list[i], l6_list[i]]

            # Process first decision point
            if d1_list[i] == 0:  # Take first arm
                ts1 = l1_list[i]
                tm1 = np.random.normal(ts1, ts1*wb)  # Add noise to timing
                tm1_ = int(np.round(tm1))
                labels[i, 0, tm1_:] = 1
                
                # Process second decision point
                if d2_list[i] == 0:  # Go up
                    ts2 = l3_list[i]
                    tm2 = np.random.normal(ts2, ts2*wb)
                    tm2_ = int(np.round(tm2))
                    labels[i, 2, time_pre_stim + tm1_ + tm2_:] = 1
                else:  # Go down
                    ts2 = l4_list[i]
                    tm2 = np.random.normal(ts2, ts2*wb)
                    tm2_ = int(np.round(tm2))
                    labels[i, 3, time_pre_stim + tm1_ + tm2_:] = 1

            else:  # Take second arm
                ts1 = l2_list[i]
                tm1 = np.random.normal(ts1, ts1*wb)
                tm1_ = int(np.round(tm1))
                labels[i, 1, tm1_:] = 1
                
                # Process second decision point
                if d2_list[i] == 0:  # Go up
                    ts2 = l5_list[i]
                    tm2 = np.random.normal(ts2, ts2*wb)
                    tm2_ = int(np.round(tm2))
                    labels[i, 2, time_pre_stim + tm1_ + tm2_:] = 1
                else:  # Go down
                    ts2 = l6_list[i]
                    tm2 = np.random.normal(ts2, ts2*wb)
                    tm2_ = int(np.round(tm2))
                    labels[i, 3, time_pre_stim + tm1_ + tm2_:] = 1

            # Ensure minimum timing values and generate time course
            tm1_ = max(tm1_, 1)
            tm2_ = max(tm2_, 1)
            in_time[i, 0, time_pre_stim:time_pre_stim+tm1_+1] = np.arange(tm1_+1)/tm1_*tm1
            in_time[i, 0, time_pre_stim+tm1_:] = tm1

            in_time[i, 1, time_pre_stim + tm1_:time_pre_stim + tm1_ + tm2_ + 1] = np.arange(tm2_+1)/tm2_*tm2
            in_time[i, 1, time_pre_stim+tm1_+tm2_:] = tm2

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Sample requested number of trials
        if ask_sample_size <= n_total:
            idx = np.random.choice(int(n_total/rep), ask_sample_size, replace=False)
        else:
            raise ValueError('Asked size larger than possible size')

        # Reshape arrays and select samples
        in_arms = np.reshape(in_arms, (int(in_arms.shape[0]/rep), rep, in_arms.shape[1], in_arms.shape[2]))
        in_time = np.reshape(in_time, (int(in_time.shape[0]/rep), rep, in_time.shape[1], in_time.shape[2]))
        labels = np.reshape(labels, (int(labels.shape[0]/rep), rep, labels.shape[1], labels.shape[2]))

        in_arms = in_arms[idx, :, :, :]
        in_time = in_time[idx, :, :, :]
        labels = labels[idx, :, :, :]

        # Flatten arrays
        in_arms = np.reshape(in_arms, (in_arms.shape[0]*rep, in_arms.shape[2], in_arms.shape[3]))
        in_time = np.reshape(in_time, (in_time.shape[0]*rep, in_time.shape[2], in_time.shape[3]))
        labels = np.reshape(labels, (labels.shape[0]*rep, labels.shape[2], labels.shape[3]))

        # Split into training and validation sets
        train_sample_size = int(n_total * (1 - self.val_size))
        idx_train = np.arange(train_sample_size)
        idx_train = np.random.choice(idx_train, len(idx_train), replace=False)
        idx_val = np.setdiff1d(np.arange(n_total), idx_train)
        idx_val = np.random.choice(idx_val, len(idx_val), replace=False)

        # Return results dictionary
        results = {
            'in_arms': in_arms,  # Arm length inputs
            'in_time': in_time,  # Timing inputs
            'labels': labels,    # Target outputs
            'idx_train': idx_train,  # Training indices
            'idx_val': idx_val   # Validation indices
        }
        return results