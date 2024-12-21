import copy

import numpy as np
import torch
import matplotlib.pyplot as plt

# this class takes parameters of the task_params, and generates the teacher input/output of the task_params

# noise as wb fraction
class hmaze_task(object):
    def __init__(self, params):
        self.time_trial = params['time_trial']
        self.time_pre_stim = params['time_pre_stim']
        # self.time_cue = params['time_cue']
        # self.grace_intv = params['grace_intv']
        self.length_arm1 = params['length_arm1']
        self.length_arm2 = params['length_arm2']
        self.length_arm3 = params['length_arm3']
        self.length_arm4 = params['length_arm4']
        self.length_arm5 = params['length_arm5']
        self.length_arm6 = params['length_arm6']
        self.velocity = params['velocity']
        self.D1 = params['D1']
        self.D2 = params['D2']
        self.ask_sample_size = params['ask_sample_size']
        self.in_size = params['in_size']
        self.out_size = params['out_size']
       # self.std = params['std']
        self.rep_per_maze = params['rep_per_maze']
        self.val_size = params['val_size']
        self.rand_seed = params['rand_seed']
        self.wb = params['wb']
        self.wb_inc = params['wb_inc']

    def __call__(self, seed=None):
        time_trial = self.time_trial
        l1 = self.length_arm1
        l2 = self.length_arm2
        l3 = self.length_arm3
        l4 = self.length_arm4
        l5 = self.length_arm5
        l6 = self.length_arm6
        # time_cue = self.time_cue
        d1 = self.D1
        d2 = self.D2
        ask_sample_size = self.ask_sample_size
        # grace_intv = self.grace_intv
        in_size = self.in_size
        out_size = self.out_size
        rep = self.rep_per_maze
        n_conditions = len(l1) * len(l2) * len(l3) * len(l4) * len(l5) * len(l6) * len(d1) * len(d2)
        n_total = n_conditions * rep
        time_pre_stim = self.time_pre_stim
        wb = self.wb
        #std = self.std


        # generate meshgrid for all inputs
        # switching between arm1 and arm2 is intentional due to function feature
        mesh = np.meshgrid(l1, l2, l3, l4, l5, l6, d1, d2, np.arange(rep))
        l1_list = mesh[0].flatten()
        l2_list = mesh[1].flatten()
        l3_list = mesh[2].flatten()
        l4_list = mesh[3].flatten()
        l5_list = mesh[4].flatten()
        l6_list = mesh[5].flatten()
        d1_list = mesh[6].flatten()
        d2_list = mesh[7].flatten()

        in_arms = np.zeros((n_total, in_size - 2, time_trial))
        in_time = np.zeros((n_total, 2, time_trial))
        labels = np.zeros((n_total, out_size, time_trial))
        for i in range(n_total):
            # arm length
            in_arms[i, 0, :] = l1_list[i]
            in_arms[i, 1, :] = l2_list[i]
            in_arms[i, 2, :] = l3_list[i]
            in_arms[i, 3, :] = l4_list[i]
            in_arms[i, 4, :] = l5_list[i]
            in_arms[i, 5, :] = l6_list[i]


            # assign ts1
            if d1_list[i] == 0:
                ts1 = l1_list[i]
                tm1 = np.random.normal(ts1, ts1*wb)
                tm1_ = int(np.round(tm1))
                labels[i, 0, tm1_:] = 1
                if d2_list[i] == 0:  # right up
                    ts2 = l3_list[i]
                    tm2 = np.random.normal(ts2, ts2*wb)
                    tm2_ = int(np.round(tm2))
                    labels[i, 2, time_pre_stim + tm1_ + tm2_:] = 1
                else:  # right down
                    ts2 = l4_list[i]
                    tm2 = np.random.normal(ts2, ts2*wb)
                    tm2_ = int(np.round(tm2))
                    labels[i, 3, time_pre_stim + tm1_ + tm2_:] = 1

            else:
                ts1 = l2_list[i]
                tm1 = np.random.normal(ts1, ts1*wb)
                tm1_ = int(np.round(tm1))
                labels[i, 1, tm1_:] = 1
                if d2_list[i] == 0:  # right up
                    ts2 = l5_list[i]
                    tm2 = np.random.normal(ts2, ts2*wb)
                    tm2_ = int(np.round(tm2))
                    labels[i, 2, time_pre_stim + tm1_ + tm2_:] = 1
                else:  # right down
                    ts2 = l6_list[i]
                    tm2 = np.random.normal(ts2, ts2*wb)
                    tm2_ = int(np.round(tm2))
                    labels[i, 3, time_pre_stim + tm1_ + tm2_:] = 1

            tm1_ = max(tm1_, 1)
            tm2_ = max(tm2_, 1)
            in_time[i, 0, time_pre_stim:time_pre_stim+tm1_+1] = np.arange(tm1_+1)/tm1_*tm1
            in_time[i, 0, time_pre_stim+tm1_:] = tm1

            in_time[i, 1, time_pre_stim + tm1_:time_pre_stim + tm1_ + tm2_ + 1] = np.arange(tm2_+1)/tm2_*tm2
            in_time[i, 1, time_pre_stim+tm1_+tm2_:] = tm2

            # output
            # labels[i, d1_list[i], time_cue:] = 1

        if seed is not None:
            np.random.seed(seed)

        if ask_sample_size <= n_total:
                        idx = np.random.choice(int(n_total/rep), ask_sample_size, replace=False)
        else:
            raise ValueError('Asked size larger than possible size')

        in_arms = np.reshape(in_arms, (int(in_arms.shape[0]/rep), rep, in_arms.shape[1], in_arms.shape[2]))
        in_time = np.reshape(in_time, (int(in_time.shape[0]/rep), rep, in_time.shape[1], in_time.shape[2]))
        labels = np.reshape(labels, (int(labels.shape[0]/rep), rep, labels.shape[1], labels.shape[2]))

        in_arms = in_arms[idx, :, :, :]
        in_time = in_time[idx, :, :, :]
        labels = labels[idx, :, :, :]

        in_arms = np.reshape(in_arms, (in_arms.shape[0]*rep, in_arms.shape[2], in_arms.shape[3]))
        in_time = np.reshape(in_time, (in_time.shape[0]*rep, in_time.shape[2], in_time.shape[3]))
        labels = np.reshape(labels, (labels.shape[0]*rep, labels.shape[2], labels.shape[3]))

        train_sample_size = int(n_total * (1 - self.val_size))
        idx_train = np.arange(train_sample_size)
        # np.random.seed(0)
        idx_train = np.random.choice(idx_train, len(idx_train), replace=False)
        idx_val = np.setdiff1d(np.arange(n_total), idx_train)
        idx_val = np.random.choice(idx_val, len(idx_val), replace=False)

        results = {
            'in_arms': in_arms,
            'in_time': in_time,
            'labels': labels,
            'idx_train': idx_train,
            'idx_val': idx_val
        }
        return results