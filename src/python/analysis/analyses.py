import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import numpy.linalg as la
import scipy.stats as stats
import torch
import copy
from sklearn.decomposition import PCA
from os.path import exists
from scipy.io import savemat
import itertools
import probablistic_models.models as p_models

def ExportActivity(params):

    def _acc_func(model_results):
        ########
        keys_batch = model_results['keys'][:, :, 25:]

        model_d1 = torch.mean(keys_batch[:, 0, :], 1) < 0.5
        model_d2 = torch.mean(keys_batch[:, 1, :], 1) < 0.5

        choices_model = np.zeros(n_trials)
        for j in range(n_trials):
            choices_model[j] = mapping_label[(int(model_d1[j]), int(model_d2[j]))]

        return choices_model

    save_path = params['save_path']
    net = params['net']
    task_data_generator = params['task_data_generator']
    label_func = eval(params['trainer_params']['acc_func'])

    trials = task_data_generator()
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    labels = trials['labels'][:, :, -1]
    n_trials = in_arms.shape[0]

    mapping_label = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }
    labels_d1 = np.argmax(labels[:, :2], 1)
    labels_d2 = np.argmax(labels[:, 2:], 1)
    labels_comb = np.zeros(n_trials)
    for i in range(n_trials):
        labels_comb[i] = mapping_label[(labels_d1[i], labels_d2[i])]

    inputs = np.concatenate((in_arms, in_time), axis=1)
    inputs = torch.from_numpy(inputs)
    model_results = net(copy.deepcopy(inputs))

    model_results['keys'] = model_results['keys'].detach().numpy()
    model_results['prs'] = model_results['prs'].detach().numpy()
    model_results['hs'] = model_results['hs'].detach().numpy()

    rnn = {
        "W_in_bias": net.W_linear_in.bias.data.numpy(),
        "W_in_weight": net.W_linear_in.weight.data.numpy(),
        "W_rec_bias": net.W_rec.bias.data.numpy(),
        "W_rec_weight": net.W_rec.weight.data.numpy(),
        "W_out_bias": net.W_key.bias.data.numpy(),
        "W_out_weight": net.W_key.weight.data.numpy(),
    }
    matfile = {
        "input": trials,
        "output": model_results,
        "rnn": rnn
    }

    savemat("joint_rnn.mat", matfile)

    pass


def Psychometric(params):
    def _acc_ext_supervised(model_results):
        outputs = model_results['outputs'].detach().numpy()
        model_d1 = np.argmax(np.mean(outputs[:, :2, 25:], 2), 1)
        model_d2 = np.argmax(np.mean(outputs[:, 2:, 25:], 2), 1)
        choices_model = np.zeros(len(n_trials))
        for j in range(len(n_trials)):
            choices_model[j] = mapping_label[(model_d1[j], model_d2[j])]
        return choices_model

    def _acc_func(model_results):
        ########
        keys_batch = model_results['keys'][:, :, 25:]

        model_d1 = torch.mean(keys_batch[:, 0, :], 1) < 0.5
        model_d2 = torch.mean(keys_batch[:, 1, :], 1) < 0.5

        choices_model = np.zeros(n_trials)
        for j in range(n_trials):
            choices_model[j] = mapping_label[(int(model_d1[j]), int(model_d2[j]))]

        return choices_model

    save_path = params['save_path']
    net = params['net']
    task_data_generator = params['task_data_generator']
    label_func = eval(params['trainer_params']['acc_func'])

    trials = task_data_generator()
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    labels = trials['labels'][:, :, -1]
    n_trials = in_arms.shape[0]

    mapping_label = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }
    labels_d1 = np.argmax(labels[:, :2], 1)
    labels_d2 = np.argmax(labels[:, 2:], 1)
    labels_comb = np.zeros(n_trials)
    for i in range(n_trials):
        labels_comb[i] = mapping_label[(labels_d1[i], labels_d2[i])]

    inputs = np.concatenate((in_arms, in_time), axis=1)
    inputs = torch.from_numpy(inputs)
    model_results = net(copy.deepcopy(inputs))

    choices_model = label_func(model_results)

    acc_each = np.zeros((4, 4))
    acc_margin0 = np.zeros(4)
    acc_margin1 = np.zeros(4)
    idx_each = np.ndarray((4, 4), dtype=object)
    idx_margin0 = np.ndarray((4), dtype=object)
    idx_margin1 = np.ndarray((4), dtype=object)
    mapping_idx = {
        3: 0,
        2: 1,
        1: 2,
        0: 3
    }
    for i in range(n_trials):
        idx0 = mapping_idx[np.abs(in_arms[i, 0, -1] - in_arms[i, 1, -1])]

        if labels_d1[i] == 0:
            idx1 = mapping_idx[np.abs(in_arms[i, 2, -1] - in_arms[i, 3, -1])]
        else:
            idx1 = mapping_idx[np.abs(in_arms[i, 4, -1] - in_arms[i, 5, -1])]

        if idx_each[idx0, idx1] is None:
            idx_each[idx0, idx1] = []
        else:
            idx_each[idx0, idx1].append(i)

        if idx_margin0[idx0] is None:
            idx_margin0[idx0] = []
        else:
            idx_margin0[idx0].append(i)

        if idx_margin1[idx1] is None:
            idx_margin1[idx1] = []
        else:
            idx_margin1[idx1].append(i)

    for i in range(4):
        acc_margin0[i] = sum(choices_model[idx_margin0[i]] == labels_comb[idx_margin0[i]]) / len(idx_margin0[i])
        acc_margin1[i] = sum(choices_model[idx_margin1[i]] == labels_comb[idx_margin1[i]]) / len(idx_margin1[i])
        for j in range(4):
            idx_ij = idx_each[i, j]
            acc_each[i, j] = sum(choices_model[idx_ij] == labels_comb[idx_ij]) / len(idx_ij)

    fig, ax = plt.subplots(1, 3, figsize=(20, 3))
    cmap = 'jet'
    h = ax[0].imshow(np.flip(acc_each, axis=1), cmap=mpl.colormaps[cmap], vmin=0, vmax=1)
    ax[0].set_xlabel('Horizontal Diff.')
    ax[0].set_ylabel('Vertical Diff.')
    ax[1].imshow(np.flip(acc_margin0[:, np.newaxis], axis=0), cmap=mpl.colormaps[cmap], vmin=0, vmax=1)
    ax[1].set_title('Horizontal marginal')
    ax[2].imshow(np.flip(acc_margin1[:, np.newaxis], axis=0), cmap=mpl.colormaps[cmap], vmin=0, vmax=1)
    ax[2].set_title('Vertical marginal')
    fig.colorbar(h, ax=ax[0])
    plt.show()


def saccade(params):
    task_data_generator = params['task_data_generator']
    net = params['net']
    trials = task_data_generator()
    in_arms = trials['in_arms']
    tm = trials['in_time']
    idx = 0
    trial1 = [4, 7, 5, 7, 6, 4, 4, 7]  # easy/easy
    trial2 = [4, 6, 6, 7, 5, 4, 6, 7]  # hierarchical
    trial3 = [6, 5, 7, 5, 4, 4, 5, 7]  # counterfactual
    # trial2 = [7, 4, 5, 6, 7, 4, 7, 6]  # easy/hard
    # trial3 = [5, 6, 7, 4, 4, 7, 6, 7]  # hard/easy
    # trial4 = [5, 6, 4, 5, 7, 6, 6, 7]  # hard/hard
    trial_list = [trial1, trial2, trial3]
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    for t in range(len(trial_list)):
        np.random.seed(3)

        trial = trial_list[t]
        mannual = True
        if mannual:
            in_arms[idx, 0, :] = trial[0]
            in_arms[idx, 1, :] = trial[1]
            in_arms[idx, 2, :] = trial[2]
            in_arms[idx, 3, :] = trial[3]
            in_arms[idx, 4, :] = trial[4]
            in_arms[idx, 5, :] = trial[5]
            tm[idx, :, :] = 0
            tm1 = trial[6]
            tm2 = trial[7]

            if True:
                tm1_ = int(np.round(tm1))
                tm2_ = int(np.round(tm2))
                tm1_ = max(tm1_, 1)
                tm2_ = max(tm2_, 1)
                tm[idx, 0, 5:5 + tm1_ + 1] = np.arange(tm1_ + 1) / tm1_ * tm1
                tm[idx, 0, 5 + tm1_:] = tm1

                tm[idx, 1, 5 + tm1_:5 + tm1_ + tm2_ + 1] = np.arange(tm2_ + 1) / tm2_ * tm2
                tm[idx, 1, 5 + tm1_ + tm2_:] = tm2

        inputs = np.concatenate((in_arms, tm), axis=1)
        # inputs[idx, 3, :]
        results = net(torch.from_numpy(np.expand_dims(inputs[idx, :, :], 0)))
        # outputs = torch.squeeze(results['outputs'])
        # outputs = outputs.detach().numpy()
        keys = torch.squeeze(results['keys'])
        keys = keys.detach().numpy()
        prs = torch.squeeze(results['prs'])
        prs = prs.detach().numpy()

        tm_real = np.squeeze(results['tm_modified'])
        # tm_real = tm_real.detach().numpy()

        labels = trials['labels']
        n_time = in_arms.shape[2]
        n_arms = in_arms.shape[1]
        colormap = 'Set1'
        cmap = plt.get_cmap(colormap, n_arms)

        # saccades
        pre = np.arange(4)

        # 4 time points after tm1 presentation
        epoch1 = np.arange(6 + tm1_, 6 + tm1_ + 2)

        # last 4 time points
        epoch2 = np.arange(tm.shape[2] - 4, tm.shape[2] - 1)

        choice1 = np.array([1 - np.mean(keys[0, epoch1]), 0])
        choice2 = np.array([1 - np.mean(keys[0, epoch2]), 0.1])
        choice3 = np.array([1 - np.mean(keys[0, epoch2]), np.mean(keys[1, epoch2])])

        point0 = [0, 0]
        point1 = [(1 - choice1[0]) * trial[0] * -1 + choice1[0] * trial[1], 0]

        point4 = [(1 - choice3[0]) * trial[0] * -1 + choice3[0] * trial[1],
                  (1 - choice3[0]) * (choice3[1] * trial[2] + (1 - choice3[1]) * trial[3] * -1) + choice3[0] * (
                          choice3[1] * trial[4] + (1 - choice3[1]) * trial[5] * -1)]
        points = np.array([point0, point1, point4])

        lgd = str(trial)
        ax1 = ax[t]
        width = 10
        color = (.4, .4, .4, .3)
        ax1.plot([0, 0], [0, 5], linewidth=width, color=(.7, .7, .7, .3))
        ax1.plot([0, -trial[0]], [0, 0], linewidth=width, color=color)
        ax1.plot([0, trial[1]], [0, 0], linewidth=width, color=color)
        ax1.plot([-trial[0], -trial[0]], [0, trial[2]], linewidth=width, color=color)
        ax1.plot([-trial[0], -trial[0]], [0, -trial[3]], linewidth=width, color=color)
        ax1.plot([trial[1], trial[1]], [0, trial[4]], linewidth=width, color=color)
        ax1.plot([trial[1], trial[1]], [0, -trial[5]], linewidth=width, color=color)

        ax1.scatter(0, 0, s=200, color=(.7, 0, 0, .7), label=lgd)
        for i in range(len(points) - 1):
            ax1.quiver(points[i, 0], points[i, 1], points[i + 1, 0] - points[i, 0], points[i + 1, 1] - points[i, 1],
                       angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=10, headlength=9,
                       color=(1, 0, 0, 0.5))

        left_end = -8
        ax1.quiver(left_end, -8, 16, 0,
                   angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=10, headlength=9,
                   color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1, left_end + 1], [-8, -7], linewidth=3, color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1 + trial[6], left_end + 1 + trial[6]], [-8, -7], linewidth=3, color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1 + trial[6] + trial[7], left_end + 1 + trial[6] + trial[7]], [-8, -7], linewidth=3,
                 color=(1, 0, 0, 0.5))

        ax1.legend()
        ax1.set_xlim([-10, 10])
        ax1.set_ylim([-10, 10])

        # ax1.legend(h1, lgd)
    plt.show()


def viz(params):
    task_data_generator = params['task_data_generator']
    net = params['net']
    trials = task_data_generator()
    in_arms = trials['in_arms']
    tm = trials['in_time']
    idx = 0
    trial1 = [4, 7, 5, 7, 6, 4, 4, 7]  # easy/easy
    trial2 = [4, 6, 6, 7, 5, 4, 6, 7]  # hierarchical
    trial3 = [5, 4, 7, 6, 4, 4, 4, 7]  # counterfactual
    # trial2 = [7, 4, 5, 6, 7, 4, 7, 6]  # easy/hard
    # trial3 = [5, 6, 7, 4, 4, 7, 6, 7]  # hard/easy
    # trial4 = [5, 6, 4, 5, 7, 6, 6, 7]  # hard/hard
    trial_list = [trial1, trial2, trial3]
    for t in range(len(trial_list)):
        trial = trial_list[t]
        mannual = True
        if mannual:
            in_arms[idx, 0, :] = trial[0]
            in_arms[idx, 1, :] = trial[1]
            in_arms[idx, 2, :] = trial[2]
            in_arms[idx, 3, :] = trial[3]
            in_arms[idx, 4, :] = trial[4]
            in_arms[idx, 5, :] = trial[5]
            tm[idx, :, :] = 0
            tm1 = trial[6]
            tm2 = trial[7]

            if True:
                tm1_ = int(np.round(tm1))
                tm2_ = int(np.round(tm2))
                tm1_ = max(tm1_, 1)
                tm2_ = max(tm2_, 1)
                tm[idx, 0, 5:5 + tm1_ + 1] = np.arange(tm1_ + 1) / tm1_ * tm1
                tm[idx, 0, 5 + tm1_:] = tm1

                tm[idx, 1, 5 + tm1_:5 + tm1_ + tm2_ + 1] = np.arange(tm2_ + 1) / tm2_ * tm2
                tm[idx, 1, 5 + tm1_ + tm2_:] = tm2

        inputs = np.concatenate((in_arms, tm), axis=1)
        # inputs[idx, 3, :]
        np.random.seed(0)
        results = net(torch.from_numpy(np.expand_dims(inputs[idx, :, :], 0)))
        # outputs = torch.squeeze(results['outputs'])
        # outputs = outputs.detach().numpy()
        keys = torch.squeeze(results['keys'])
        keys = keys.detach().numpy()
        prs = torch.squeeze(results['prs'])
        prs = prs.detach().numpy()

        tm_real = np.squeeze(results['tm_modified'])
        # tm_real = tm_real.detach().numpy()

        labels = trials['labels']
        n_time = in_arms.shape[2]
        n_arms = in_arms.shape[1]
        colormap = 'Set1'
        cmap = plt.get_cmap(colormap, n_arms)
        colors = cmap(np.arange(0, cmap.N))

        fig, ax = plt.subplots(3, 1)
        x = np.arange(n_time)

        # input primary arm lengths
        handles = []
        lgd = ('Arm 0: ' + str(in_arms[idx, 0, 0]), 'Arm 1: ' + str(in_arms[idx, 1, 0]), 'Tm 1: '
               + str(np.max(tm[idx, 0, :])))
        for i in range(2):
            h_i, = ax[0].plot(x, in_arms[idx, i, :], color=colors[i, :])
            handles.append(h_i)
        h_i, = ax[0].plot(x, tm_real[0, :], linestyle='--')
        handles.append(h_i)
        ax[0].set_ylim([-1, 15])
        ax[0].set_ylabel('Primary arms')
        ax[0].legend(handles, lgd)

        # input secondary arm lengths
        handles = []
        lgd = (
            'Arm 2: ' + str(in_arms[idx, 2, 0]), 'Arm 3: ' + str(in_arms[idx, 3, 0]),
            'Arm 4: ' + str(in_arms[idx, 4, 0]),
            'Arm 5: ' + str(in_arms[idx, 5, 0]), 'Tm2: ' + str(np.max(tm[idx, 1, :])))
        for i in range(2, 6):
            h_i, = ax[1].plot(x, in_arms[idx, i, :], color=colors[i, :])
            handles.append(h_i)
        h_i, = ax[1].plot(x, tm_real[1, :], linestyle='--')
        handles.append(h_i)
        ax[1].set_ylim([-1, 15])
        ax[1].set_ylabel('Secondary arms')
        ax[1].legend(handles, lgd)

        # keys
        keys = keys[:, 1:]
        x = x[1:]
        handles = []
        for i in range(2):
            h_i, = ax[2].plot(x, keys[i, :], color=colors[i, :])
            handles.append(h_i)
        ax[2].set_ylabel('Keys')
        lgd = ('Horizontal', 'Vertical')
        ax[2].legend(handles, lgd)

        # saccades
        pre = np.arange(4)

        # 4 time points after tm1 presentation
        epoch1 = np.arange(6 + tm1_, 6 + tm1_ + 2)

        # last 4 time points
        epoch2 = np.arange(tm.shape[2] - 4, tm.shape[2] - 1)

        choice1 = np.array([1 - np.mean(keys[0, epoch1]), 0])
        choice2 = np.array([1 - np.mean(keys[0, epoch2]), 0.1])
        choice3 = np.array([1 - np.mean(keys[0, epoch2]), np.mean(keys[1, epoch2])])
        y1 = 0.2
        y2 = -0.2
        point0 = [0, 0]
        point1 = [(1 - choice1[0]) * trial[0] * -1 + choice1[0] * trial[1], 0]
        # point2 = [(1 - choice1[0]) * trial[0] * -1 + choice1[0] * trial[1], y2]
        # point3 = [(1 - choice2[0]) * trial[0] * -1 + choice2[0] * trial[1], ]
        point4 = [(1 - choice3[0]) * trial[0] * -1 + choice3[0] * trial[1],
                  (1 - choice3[0]) * (choice3[1] * trial[2] + (1 - choice3[1]) * trial[3] * -1) + choice3[0] * (
                          choice3[1] * trial[4] + (1 - choice3[1]) * trial[5] * -1)]
        points = np.array([point0, point1, point4])
        # cmap = plt.get_cmap('Pastel1', 8)
        # colors = cmap(np.arange(0, cmap.N))
        epoch_colors = [[.5, .5, .5], colors[0, :3], colors[1, :3], colors[3, :3]]
        handles = []
        lgd = str(trial)
        fig1, ax1 = plt.subplots()
        width = 10
        color = (.4, .4, .4, .3)
        ax1.plot([0, 0], [0, 5], linewidth=width, color=(.7, .7, .7, .3))
        ax1.plot([0, -trial[0]], [0, 0], linewidth=width, color=color)
        ax1.plot([0, trial[1]], [0, 0], linewidth=width, color=color)
        ax1.plot([-trial[0], -trial[0]], [0, trial[2]], linewidth=width, color=color)
        ax1.plot([-trial[0], -trial[0]], [0, -trial[3]], linewidth=width, color=color)
        ax1.plot([trial[1], trial[1]], [0, trial[4]], linewidth=width, color=color)
        ax1.plot([trial[1], trial[1]], [0, -trial[5]], linewidth=width, color=color)

        ax1.scatter(0, 0, s=200, color=(.7, 0, 0, .7), label=lgd)
        for i in range(len(points) - 1):
            ax1.quiver(points[i, 0], points[i, 1], points[i + 1, 0] - points[i, 0], points[i + 1, 1] - points[i, 1],
                       angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=10, headlength=9,
                       color=(1, 0, 0, 0.5))

        left_end = -8
        ax1.quiver(left_end, -8, 16, 0,
                   angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=10, headlength=9,
                   color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1, left_end + 1], [-8, -7], linewidth=3, color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1 + trial[6], left_end + 1 + trial[6]], [-8, -7], linewidth=3, color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1 + trial[6] + trial[7], left_end + 1 + trial[6] + trial[7]], [-8, -7], linewidth=3,
                 color=(1, 0, 0, 0.5))

        ax1.legend()
        ax1.set_xlim([-10, 10])
        ax1.set_ylim([-10, 10])

        # ax1.legend(h1, lgd)
        plt.show()
        a = 1


# def euc_distance(P, Q):
#     '''
#     p is the true distribution
#     q is the model distribution
#     '''
#     n_bootstrap = P.shape[0]

#     result = np.zeros((n_bootstrap))
#     for ib in range(n_bootstrap):
#         p = P[ib].flatten()
#         q = Q[ib].flatten()
#         euc = []
#         for i in range(len(p)):
#             if p[i] is None:
#                 continue
#             p_v = np.zeros(4)
#             q_v = np.zeros(4)
#             for choice in range(4):
#                 p_v[choice] = (np.array(p[i])==choice).sum()/len(p[i])
#                 q_v[choice] = (np.array(q[i])==choice).sum()/len(q[i])
#             euc.append(np.linalg.norm(p_v-q_v,ord=2))
#         result[ib] = np.mean(euc)
#     return result

def KL_divergence(P, Q):
    '''
    p is the true distribution
    q is the model distribution
    '''
    n_bootstrap = P.shape[0]
    eps=1e-2
    result = np.zeros((n_bootstrap))
    for ib in range(n_bootstrap):
        p = P[ib]
        q = Q[ib]
        div = []
        for i in range(p.shape[0]):
            p_v = p[i,:]
            q_v = q[i,:]

            p_v = (p_v + eps)/(p_v + eps).sum()
            q_v = (q_v + eps)/(q_v + eps).sum()
            div.append(np.sum(p_v * np.log(p_v / q_v)))
        result[ib] = np.mean(div)
    return result





def AnalysisMatchModelStat(params):
    def _acc_ext_supervised(model_results):
        outputs = model_results['outputs'].detach().numpy()
        model_d1 = np.argmax(np.mean(outputs[:, :2, 25:], 2), 1)
        model_d2 = np.argmax(np.mean(outputs[:, 2:, 25:], 2), 1)
        n_trials = model_results['keys'].shape[0]
        choices_model = np.zeros(n_trials)
        for j in range(n_trials):
            choices_model[j] = mapping[(model_d1[j], model_d2[j])]
        return choices_model
    
    
    def _acc_func(model_results):
        ########
        keys_batch = model_results['keys'][:, :, 25:]

        d1 = torch.mean(keys_batch[:, 0, :], 1) < 0.5
        d2 = torch.mean(keys_batch[:, 1, :], 1) < 0.5
        n_trials = model_results['keys'].shape[0]

        choices_model = np.zeros(n_trials)
        for j in range(n_trials):
            choices_model[j] = mapping[(int(d1[j]), int(d2[j]))]

        return choices_model

    save_path = params['save_path']
    net = params['net']
    task_data_generator = params['task_data_generator']
    label_func = eval(params['trainer_params']['acc_func'])
    n_bootstrap = params['n_bootstrap']
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }
    wb = task_data_generator.wb
    wb_inc = task_data_generator.wb_inc
    # select the best switch threshold of counterfactual model on training set, then run 10 iterations
    
    thresh_range = [0, 1]
    n_search = 5

    emission_c_file = save_path.parents[0] / (str(np.around(wb_inc, decimals=3))+'.ems')
    if emission_c_file.exists():
        results = torch.load(emission_c_file)
        emission_c_dict = results['emission_c']
    else:
        eps=1e-3

        def _loss_func(trials, wb, wb_inc, threshold):
            emission = p_models.counterfactual_emission(trials, wb, wb_inc, threshold=threshold, n_sim=1000)
            labels = trials['labels']
            result = -np.mean(np.log(emission[np.arange(len(emission)), labels.astype(int)] + eps))
            return result
        task_data_generator.rep_per_maze = 1
        trials = task_data_generator()
        n_conditions = trials['in_arms'].shape[0]
        labels_d1 = np.argmax(trials['labels'][:, :2, -1], 1)
        labels_d2 = np.argmax(trials['labels'][:, 2:, -1], 1)
        labels_comb = np.zeros(n_conditions)

        in_arms = trials['in_arms']
        in_time = trials['in_time']
        for i in range(n_conditions):
            labels_comb[i] = mapping[(labels_d1[i], labels_d2[i])]
        trials = {
            'in_arms': in_arms,
            'in_time': in_time,
            'labels': labels_comb
        }

        # perform search
        il = thresh_range[0]
        ir = thresh_range[1]
        im = (il + ir) / 2
        m = _loss_func(trials, wb, wb_inc, threshold=im)
        radius = (ir - il) / 4
        counter = 0
        while counter <= n_search: 
            il = im - radius
            ir = im + radius
            l = _loss_func(trials, wb, wb_inc, threshold=il)
            r = _loss_func(trials, wb, wb_inc, threshold=ir)

            idx = np.argmin([l,m,r])
            im = [il, im, ir][idx]
            m = [l, m, r][idx]

            radius = radius / 2
            counter += 1
            print(counter, im, m)

        
        thresh_c = im

        emission_c_dict = {}
        emission_raw = p_models.counterfactual_emission(trials, wb, wb_inc, threshold=thresh_c, n_sim=1000)
        n_conditions = trials['in_arms'].shape[0]
        for i in range(n_conditions):
            arms = trials['in_arms'][i, :, -1]
            label= mapping[(labels_d1[i], labels_d2[i])]
            key = arms.astype(int).tolist()+[label]
            key = tuple(key)
            emission_c_dict[key] = emission_raw[i]


        results = {
            'emission_c': emission_c_dict
        }
        torch.save(results, emission_c_file)


    emission_file = save_path.parents[1] / 'emission.dict'
    if emission_file.exists():
        results = torch.load(emission_file)
        emission_o_dict = results['emission_o']
        emission_p_dict = results['emission_p']
        emission_h_dict = results['emission_h']
    else:
        # select the best switch threshold of counterfactual model on training set, then run 10 iterations
        # for all models on the validation set
        task_data_generator.rep_per_maze = 1
        trials = task_data_generator()
        n_conditions = trials['in_arms'].shape[0]
        labels_d1 = np.argmax(trials['labels'][:, :2, -1], 1)
        labels_d2 = np.argmax(trials['labels'][:, 2:, -1], 1)
        labels_comb = np.zeros(n_conditions)

        in_arms = trials['in_arms']
        in_time = trials['in_time']
        for i in range(n_conditions):
            labels_comb[i] = mapping[(labels_d1[i], labels_d2[i])]
        trials = {
            'in_arms': in_arms,
            'in_time': in_time,
            'labels': labels_comb
        }
        emission_o = p_models.optimal_emission(trials, wb=0.15, n_sim=10000)
        emission_p = p_models.postdictive_emission(trials, wb=0.15, n_sim=10000)
        emission_h = p_models.hierarchy_emission(trials, wb=0.15, n_sim=10000)

        emission_o_dict = {}
        emission_p_dict = {}
        emission_h_dict = {}
        for i in range(n_conditions):
            arms = trials['in_arms'][i, :, -1]
            label= mapping[(labels_d1[i], labels_d2[i])]
            key = arms.astype(int).tolist()+[label]
            key = tuple(key)
            emission_o_dict[key] = emission_o[i]
            emission_p_dict[key] = emission_p[i]
            emission_h_dict[key] = emission_h[i]

        # save
        results = {
            'emission_o': emission_o_dict,
            'emission_p': emission_p_dict,
            'emission_h': emission_h_dict
        }

        torch.save(results, 'emission.dict')
    
    metric_all = np.zeros((n_bootstrap, 4))  # o, h, p, c

    # if there is emission dictionary

    for it in range(n_bootstrap):
        trials = task_data_generator(seed=it)
        # idx_val = trials['idx_val']
        idx_val = np.arange(len(trials['in_arms']))
        in_arms_val = trials['in_arms'][idx_val, :, :]
        in_time_val = trials['in_time'][idx_val, :, :]
        labels_val = trials['labels'][idx_val, :, -1]


        labels_val_d1 = np.argmax(labels_val[:, :2], 1)
        labels_val_d2 = np.argmax(labels_val[:, 2:], 1)
        labels_val_comb = np.zeros(len(idx_val))
        for i in range(len(idx_val)):
            labels_val_comb[i] = mapping[(labels_val_d1[i], labels_val_d2[i])]
        # labels_val = np.argmax(labels_val, axis=1)

        mazes = []
        for i in range(in_arms_val.shape[0]):
            mazes.append(tuple(in_arms_val[i, :, -1].astype(int).tolist() + [int(labels_val_comb[i])]))

        inputs = np.concatenate((in_arms_val, in_time_val), axis=1)
        inputs = torch.from_numpy(inputs)
        model_results = net(copy.deepcopy(inputs))

        choices_model = label_func(model_results)

        eps=1e-2
        for i in range(len(mazes)):
            maze = tuple(mazes[i])
            choice_rnn = int(choices_model[i])

            metric_all[it, 0] -= np.log(emission_o_dict[maze][choice_rnn] + eps)
            metric_all[it, 1] -= np.log(emission_h_dict[maze][choice_rnn] + eps)
            metric_all[it, 2] -= np.log(emission_p_dict[maze][choice_rnn] + eps)
            metric_all[it, 3] -= np.log(emission_c_dict[maze][choice_rnn] + eps)

        metric_all[it, :] /= len(mazes)


    results = {
        'nll': metric_all
    }

    torch.save(results, str(save_path) + '.nll')

    return results



def AnalysisWMNoiseSweep(params):
    def perf_label(model_results):
        outputs = model_results['outputs'].detach().numpy()
        model_d1 = np.argmax(np.mean(outputs[:, :2, 25:], 2), 1)
        model_d2 = np.argmax(np.mean(outputs[:, 2:, 25:], 2), 1)
        choices_model = np.zeros(len(idx_val))
        for j in range(len(idx_val)):
            choices_model[j] = mapping[(model_d1[j], model_d2[j])]
        return choices_model

    def _acc_func(model_results):
        ########
        keys_batch = model_results['keys'][:, :, 25:]

        d1 = torch.mean(keys_batch[:, 0, :], 1) < 0.5
        d2 = torch.mean(keys_batch[:, 1, :], 1) < 0.5
        n_trials = model_results['keys'].shape[0]

        choices_model = np.zeros(n_trials)
        for j in range(n_trials):
            choices_model[j] = mapping[(int(d1[j]), int(d2[j]))]

        return choices_model

    save_path = params['save_path']
    net = params['net']
    task_data_generator = params['task_data_generator']
    label_func = eval(params['trainer_params']['acc_func'])
    wb = task_data_generator.wb
    wb_inc = task_data_generator.wb_inc
    iteration = 10
    # select the best switch threshold of counterfactual model on training set, then run 10 iterations
    # for all models on the validation set
    thresh = np.arange(0.01, 0.2, 0.01)
    trials = task_data_generator()
    idx_train = trials['idx_train']
    in_arms_train = trials['in_arms'][idx_train, :, :]
    in_time_train = trials['in_time'][idx_train, :, :]
    labels_train = trials['labels'][idx_train, :, -1]
    trials_train = {
        'in_arms': in_arms_train,
        'in_time': in_time_train,
        'labels': labels_train
    }
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }
    labels_train_d1 = np.argmax(labels_train[:, :2], 1)
    labels_train_d2 = np.argmax(labels_train[:, 2:], 1)
    labels_train_comb = np.zeros(len(idx_train))
    for i in range(len(idx_train)):
        labels_train_comb[i] = mapping[(labels_train_d1[i], labels_train_d2[i])]

    choices_c = np.ndarray(len(thresh), dtype=object)
    acc_c = np.zeros(len(thresh))

    for i, th in enumerate(thresh):
        choices_c[i] = p_models.counterfactual(trials_train, wb, wb_inc, threshold=th)
        acc_c[i] = sum(labels_train_comb == choices_c[i]) / len(choices_c[i])
        # print(i)

    thresh_c = thresh[np.argmax(acc_c)]

    euc_all = np.zeros((iteration, 4))  # o, h, p, c
    for it in range(iteration):
        trials = task_data_generator()
        idx_val = trials['idx_val']
        in_arms_val = trials['in_arms'][idx_val, :, :]
        in_time_val = trials['in_time'][idx_val, :, :]
        labels_val = trials['labels'][idx_val, :, -1]
        trials_val = {
            'in_arms': in_arms_val,
            'in_time': in_time_val,
            'labels': labels_val
        }

        labels_val_d1 = np.argmax(labels_val[:, :2], 1)
        labels_val_d2 = np.argmax(labels_val[:, 2:], 1)
        labels_val_comb = np.zeros(len(idx_val))
        for i in range(len(idx_val)):
            labels_val_comb[i] = mapping[(labels_val_d1[i], labels_val_d2[i])]
        # labels_val = np.argmax(labels_val, axis=1)

        choices_p = p_models.postdictive(trials_val, wb)
        choices_o = p_models.optimal(trials_val, wb)
        choices_h = p_models.hierarchy(trials_val, wb)
        choices_c = p_models.counterfactual(trials_val, wb, wb_inc, threshold=thresh_c)

        inputs = np.concatenate((in_arms_val, in_time_val), axis=1)
        inputs = torch.from_numpy(inputs)
        model_results = net(copy.deepcopy(inputs))

        choices_model = label_func(model_results)


        acc_c = sum(labels_val_comb == choices_c) / len(choices_c)


        # coordinate metric
        n_length = len(np.unique(in_arms_val[:, 0, 0]))
        n_diffs = n_length * 2 - 1
        # difference between primary arms, left side, right side

        indices = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)
        vector_o = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)
        vector_h = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)
        vector_p = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)

        vector_c = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)
        vector_model = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)

        # map to the other vertical arm
        mapping10 = {
            0: 1,
            1: 0,
            2: 3,
            3: 2
        }

        # map to the other side of the maze
        mapping0 = {
            0: [2, 3],
            1: [2, 3],
            2: [0, 1],
            3: [0, 1]
        }

        # calculate the vector for each maze condition for each strategy
        for i in range(len(in_arms_val)):
            arms_i = in_arms_val[i, :, 0]
            tm2_i = in_time_val[i, 1, -1]
            diff1 = int(arms_i[0] - arms_i[1] + n_length - 1)
            diff2 = int(arms_i[2] - arms_i[3] + n_length - 1)
            diff3 = int(arms_i[4] - arms_i[5] + n_length - 1)

            # for each trial, needs to figure out the D11, D10, D01, D00 arms

            label_i = labels_val_comb[i]
            order = [label_i, mapping10[label_i]]  # 11, 10 arm

            arms0 = np.array(mapping0[label_i])  # incorrect primary arm
            error0 = np.abs(arms_i[arms0 + 2] - tm2_i)
            arm01_idx = np.random.choice(np.where(error0 == np.min(error0))[0], 1)  # one line fuzzy max
            arm01 = arms0[arm01_idx][0]
            arm00 = mapping10[arm01]
            order.append(arm01)
            order.append(arm00)
            order = np.array(order)

            # calculate the mean vector in each cell
            if vector_o[diff1, diff2, diff3] is None:
                vector_o[diff1, diff2, diff3] = [np.where(order == choices_o[i])[0][0]]
            else:
                vector_o[diff1, diff2, diff3].append(np.where(order == choices_o[i])[0][0])

            if vector_h[diff1, diff2, diff3] is None:
                vector_h[diff1, diff2, diff3] = [np.where(order == choices_h[i])[0][0]]
            else:
                vector_h[diff1, diff2, diff3].append(np.where(order == choices_h[i])[0][0])

            if vector_p[diff1, diff2, diff3] is None:
                vector_p[diff1, diff2, diff3] = [np.where(order == choices_p[i])[0][0]]
            else:
                vector_p[diff1, diff2, diff3].append(np.where(order == choices_p[i])[0][0])

            if vector_c[diff1, diff2, diff3] is None:
                vector_c[diff1, diff2, diff3] = [np.where(order == choices_c[i])[0][0]]
            else:
                vector_c[diff1, diff2, diff3].append(np.where(order == choices_c[i])[0][0])

            if vector_model[diff1, diff2, diff3] is None:
                vector_model[diff1, diff2, diff3] = [np.where(order == choices_model[i])[0][0]]
            else:
                vector_model[diff1, diff2, diff3].append(np.where(order == choices_model[i])[0][0])

        # compare vector means between model and three strategies
        vector_o = vector_o.flatten()
        vector_h = vector_h.flatten()
        vector_p = vector_p.flatten()

        vector_c = vector_c.flatten()
        vector_model = vector_model.flatten()
        euc_o = []
        euc_h = []
        euc_p = []
        euc_c = []
        for i in range(len(vector_model)):
            if vector_model[i] is not None:
                v_m = np.array(vector_model[i])
                v_m1 = np.zeros((len(v_m), 4))
                v_m1[np.arange(v_m1.shape[0]), v_m] = 1
                v_m1 = np.mean(v_m1, axis=0)

                v_o = np.array(vector_o[i])
                v_o1 = np.zeros((len(v_o), 4))
                v_o1[np.arange(v_o1.shape[0]), v_o] = 1
                v_o1 = np.mean(v_o1, axis=0)
                euc_o.append(v_o1 - v_m1)

                v_h = np.array(vector_h[i])
                v_h1 = np.zeros((len(v_h), 4))
                v_h1[np.arange(v_h1.shape[0]), v_h] = 1
                v_h1 = np.mean(v_h1, axis=0)
                euc_h.append(v_h1 - v_m1)

                v_p = np.array(vector_p[i])
                v_p1 = np.zeros((len(v_p), 4))
                v_p1[np.arange(v_p1.shape[0]), v_p] = 1
                v_p1 = np.mean(v_p1, axis=0)
                euc_p.append(v_p1 - v_m1)

                v_c = np.array(vector_c[i])
                v_c1 = np.zeros((len(v_c), 4))
                v_c1[np.arange(v_c1.shape[0]), v_c] = 1
                v_c1 = np.mean(v_c1, axis=0)
                euc_c.append(v_c1 - v_m1)

        euc_all[it, 0] = np.mean(la.norm(np.array(euc_o)[:, :], ord=2, axis=1))
        euc_all[it, 1] = np.mean(la.norm(np.array(euc_h)[:, :], ord=2, axis=1))
        euc_all[it, 2] = np.mean(la.norm(np.array(euc_p)[:, :], ord=2, axis=1))

        # for i in range(len(thresh)):
        #     euc_c[i] = np.mean(la.norm(np.array(euc_c[i])[:, :], ord=2, axis=1))
        euc_all[it, 3] = np.mean(la.norm(np.array(euc_c)[:, :], ord=2, axis=1))

        # # filter trials with correct answer
        #
        # idx_good = np.where(np.logical_and(np.logical_xor(choices_model <= 1, choices_o <= 1),
        #                                    in_arms_val[:, 0, -1] != in_arms_val[:, 1, -1]))
        # idx_c = np.where(np.logical_and(choices_model == choices_c[1], choices_model != choices_o))[0]
        # idx_o = np.where(np.logical_and(choices_model != choices_c[1], choices_model == choices_o))[0]
        # idx_h = np.where(np.logical_and(choices_model == choices_h, choices_model != choices_c[1]))[0]
        #
        # idx_goodc = np.intersect1d(idx_good, idx_c)
        # viz(net, inputs, idx_goodc[0])

        print(it)

    results = {
        'euc_dist': euc_all,
    }

    torch.save(results, str(save_path) + '.euc')

    return results


def SwitchFrequency(params):
    save_path = params['save_path']
    net = params['net']
    task_data_generator = params['task_data_generator']

    freq = np.zeros(4)  # 4 differences

    trials = task_data_generator()
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    labels = trials['labels'][:, :, -1]

    inputs = np.concatenate((in_arms, in_time), axis=1)
    inputs = torch.from_numpy(inputs)
    model_results = net(copy.deepcopy(inputs))
    keys = model_results['keys'].detach().numpy()
    # only count switch after tm1
    idx_tm = np.argmax(in_time, axis=2)
    n_trials = inputs.shape[0]
    switch_counts = np.zeros(n_trials)
    for n in range(n_trials):
        switches = keys[n, 0, idx_tm[n, 0]:] < 0.5
        switch_counts[n] = np.sum(np.logical_xor(np.roll(switches, 1), switches)[1:])

    diff_h = abs(inputs[:, 0, 0] - inputs[:, 1, 0]).detach().numpy()
    for diff in range(4):
        freq[diff] = switch_counts[diff_h == diff].mean()

    results = {
        'freq': freq,
    }

    torch.save(results, str(save_path) + '.sf')
    return results


def AnalysisWholeTraj(params):
    net = params['net']
    task_data_generator = params['task_data_generator']

    idx = params['task_samples']['idx_val']

    in_arms = params['task_samples']['in_arms'][idx, :, :]
    in_time = params['task_samples']['in_time'][idx, :, :]
    labels = params['task_samples']['labels'][idx, :, -1]
    trials = {
        'in_arms': in_arms,
        'in_time': in_time,
        'labels': labels
    }
    choices_o = p_models.optimalStd(trials)
    choices_c = p_models.counterfactualStd(trials, threshold=0.01)

    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }
    labels_d1 = np.argmax(labels[:, :2], 1)
    labels_d2 = np.argmax(labels[:, 2:], 1)
    labels = np.zeros(len(idx))
    for i in range(len(idx)):
        labels[i] = mapping[(labels_d1[i], labels_d2[i])]
    # labels_val = np.argmax(labels_val, axis=1)

    inputs = np.concatenate((in_arms, in_time), axis=1)
    inputs = torch.from_numpy(inputs)
    model_results = net(copy.deepcopy(inputs))

    hs = model_results['hs'].detach().numpy()
    keys = model_results['keys'].detach().numpy()
    prs_batch = model_results['prs'][:, :, 25:]
    keys_batch = model_results['keys'][:, :, 25:]

    # PCA on all data
    hs_pca = np.transpose(hs, (0, 2, 1))
    hs_pca = np.reshape(hs_pca, (hs_pca.shape[0] * hs_pca.shape[1], hs_pca.shape[2]))
    pca = PCA()
    pca.fit(hs_pca)
    pc3 = pca.components_[:3, :]
    var_expl = np.cumsum(pca.explained_variance_ratio_)

    plot_var = 0
    if plot_var:
        figure, ax = plt.subplots()
        x = np.arange(1, len(var_expl) + 1)
        ax.plot(x, var_expl)
        ax.set_xlabel('num of PCs')
        ax.set_ylabel('var. Expl.')
        plt.show()

    hs_pc3 = np.dot(hs_pca, pc3.T)
    hs_pc3 = np.reshape(hs_pc3, (hs.shape[0], hs.shape[2], hs_pc3.shape[1]))

    model_d1 = torch.argmax(torch.mean(keys_batch[:, :, :], 2), 1).detach().numpy()
    model_d2 = torch.argmin(torch.mean(torch.abs(prs_batch[:, 3:5, :]), 2), 1).detach().numpy()
    choices_model = np.zeros(len(idx))
    for i in range(len(idx)):
        choices_model[i] = mapping[(model_d1[i], model_d2[i])]

    idx_good = np.where(np.logical_and(np.logical_xor(choices_model <= 1, choices_o <= 1),
                                       in_arms[:, 0, -1] != in_arms[:, 1, -1]))
    idx_c = np.where(np.logical_and(choices_model == choices_c, choices_model != choices_o))[0]
    idx_goodc = np.intersect1d(idx_good, idx_c)

    n_bins = 10
    lim = [-3, 3]
    idx_table = np.ndarray(n_bins, dtype=object)
    for i in range(len(idx)):
        tm1 = in_time[i, 0, -1]
        diff_pri = np.abs(in_arms[i, 0, 0] - tm1) - np.abs(in_arms[i, 1, 0] - tm1)
        checker = int((diff_pri - lim[0]) / ((lim[1] - lim[0]) / n_bins))
        if checker == n_bins:
            checker -= 1

        if idx_table[checker] is None:
            idx_table[checker] = [i]
        else:
            idx_table[checker].append(i)

    colormap = 'jet'
    cmap = plt.get_cmap(colormap, n_bins)
    colors = cmap(np.arange(0, cmap.N))
    colors = np.flip(colors, axis=0)

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    lgd = []
    handles = []
    for i in range(n_bins):  # [0,2,7,9]:#
        idx_i = np.array(idx_table[i])
        idx_i = np.random.choice(idx_i, min(int(len(idx_i) / 5), 10), replace=False)

        data_i = hs_pc3[idx_i, :, :]
        keys_i = keys[idx_i, :, :]

        for j in range(len(idx_i)):
            h, = ax.plot(data_i[j, :, 0], data_i[j, :, 1], data_i[j, :, 2], color=colors[i, :], linewidth=2, alpha=0.9)

            if keys_i[j, 0, -1] > 0.5:
                c = [1, 0, 0]
            else:
                c = [0, 0, 1]
        lgd.append(np.round(lim[0] + i * ((lim[1] - lim[0]) / n_bins), 2))
        handles.append(h)

        idx_left = keys_i[:, 0, -1] >= 0.5
        idx_right = keys_i[:, 0, -1] < 0.5
        ax.scatter(data_i[idx_left, -1, 0], data_i[idx_left, -1, 1], data_i[idx_left, -1, 2], color=c, s=20, alpha=0.8)
        ax.scatter(data_i[idx_right, -1, 0], data_i[idx_right, -1, 1], data_i[idx_right, -1, 2], color=c, s=20,
                   alpha=0.8)

        idx_tm1 = np.argmax(in_time[idx_i, 0, :], 1) - 1
        idx_tm2 = np.argmax(in_time[idx_i, 1, :], 1) - 1
        ax.scatter(hs_pc3[idx_i, idx_tm1, 0], hs_pc3[idx_i, idx_tm1, 1], hs_pc3[idx_i, idx_tm1, 2], color=[0, 0, 0],
                   s=10, alpha=0.8, marker='<')
        ax.scatter(hs_pc3[idx_i, idx_tm2, 0], hs_pc3[idx_i, idx_tm2, 1], hs_pc3[idx_i, idx_tm2, 2], color=[0, 0, 0],
                   s=10, alpha=0.8, marker='8')

    ax.legend(handles, lgd)

    idx_table_c = np.ndarray(n_bins, dtype=object)
    for i in idx_goodc:
        tm1 = in_time[i, 0, -1]
        diff_pri = np.abs(in_arms[i, 0, 0] - tm1) - np.abs(in_arms[i, 1, 0] - tm1)
        checker = int((diff_pri - lim[0]) / ((lim[1] - lim[0]) / 10))
        if checker == n_bins:
            checker -= 1

        if idx_table_c[checker] is None:
            idx_table_c[checker] = [i]
        else:
            idx_table_c[checker].append(i)

    ax = fig.add_subplot(1, 2, 2, projection='3d')
    lgd = []
    handles = []
    for i in range(n_bins):
        if idx_table_c[i] is None:
            continue
        # [0,2,7,9]:#
        idx_i = np.array(idx_table_c[i])
        # idx_i = np.random.choice(idx_i, min(int(len(idx_i) / 1), 100), replace=False)

        data_i = hs_pc3[idx_i, :, :]
        keys_i = keys[idx_i, :, :]
        for j in range(len(idx_i)):
            h, = ax.plot(data_i[j, :, 0], data_i[j, :, 1], data_i[j, :, 2], color=colors[i, :], linewidth=2, alpha=0.9)
            if keys_i[j, 0, -1] > 0.5:
                c = [1, 0, 0]
            else:
                c = [0, 0, 1]

        lgd.append(np.round(lim[0] + i * ((lim[1] - lim[0]) / n_bins), 2))
        handles.append(h)

        idx_left = keys_i[:, 0, -1] >= 0.5
        idx_right = keys_i[:, 0, -1] < 0.5
        ax.scatter(data_i[idx_left, -1, 0], data_i[idx_left, -1, 1], data_i[idx_left, -1, 2], color=c, s=20, alpha=0.8)
        ax.scatter(data_i[idx_right, -1, 0], data_i[idx_right, -1, 1], data_i[idx_right, -1, 2], color=c, s=20,
                   alpha=0.8)

        idx_tm1 = np.argmax(in_time[idx_i, 0, :], 1) - 1
        idx_tm2 = np.argmax(in_time[idx_i, 1, :], 1) - 1
        ax.scatter(hs_pc3[idx_i, idx_tm1, 0], hs_pc3[idx_i, idx_tm1, 1], hs_pc3[idx_i, idx_tm1, 2], color=[0, 0, 0],
                   s=10, alpha=0.8, marker='<')
        ax.scatter(hs_pc3[idx_i, idx_tm2, 0], hs_pc3[idx_i, idx_tm2, 1], hs_pc3[idx_i, idx_tm2, 2], color=[0, 0, 0],
                   s=10, alpha=0.8, marker='8')
    ax.legend(handles, lgd)
    plt.show()


def AnalysisTmDynamics(params):
    net = params['net']
    task_data_generator = params['task_data_generator']

    idx = params['task_samples']['idx_val']

    in_arms = params['task_samples']['in_arms'][idx, :, :]
    in_time = params['task_samples']['in_time'][idx, :, :]
    labels = params['task_samples']['labels'][idx, :, -1]
    trials = {
        'in_arms': in_arms,
        'in_time': in_time,
        'labels': labels
    }
    choices_o = p_models.optimalStd(trials)
    choices_c = p_models.counterfactualStd(trials, threshold=0.01)

    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }
    mapping1 = {
        0: [2, 3],
        1: [4, 5]
    }
    labels_d1 = np.argmax(labels[:, :2], 1)
    labels_d2 = np.argmax(labels[:, 2:], 1)
    labels = np.zeros(len(idx))
    for i in range(len(idx)):
        labels[i] = mapping[(labels_d1[i], labels_d2[i])]
    # labels_val = np.argmax(labels_val, axis=1)

    inputs = np.concatenate((in_arms, in_time), axis=1)
    inputs = torch.from_numpy(inputs)
    model_results = net(copy.deepcopy(inputs))

    hs = model_results['hs'].detach().numpy()
    keys = model_results['keys'].detach().numpy()
    prs_batch = model_results['prs'][:, :, 25:]
    keys_batch = model_results['keys'][:, :, 25:]

    # PCA on all data
    hs_pca = np.transpose(hs, (0, 2, 1))
    hs_pca = np.reshape(hs_pca, (hs_pca.shape[0] * hs_pca.shape[1], hs_pca.shape[2]))
    pca = PCA()
    pca.fit(hs_pca)
    pc3 = pca.components_[:3, :]
    var_expl = np.cumsum(pca.explained_variance_ratio_)

    plot_var = 0
    if plot_var:
        figure, ax = plt.subplots()
        x = np.arange(1, len(var_expl) + 1)
        ax.plot(x, var_expl)
        ax.set_xlabel('num of PCs')
        ax.set_ylabel('var. Expl.')
        plt.show()

    hs_pc3 = np.dot(hs_pca, pc3.T)
    hs_pc3 = np.reshape(hs_pc3, (hs.shape[0], hs.shape[2], hs_pc3.shape[1]))

    model_d1 = torch.argmax(torch.mean(keys_batch[:, :, :], 2), 1).detach().numpy()
    model_d2 = torch.argmin(torch.mean(torch.abs(prs_batch[:, 3:5, :]), 2), 1).detach().numpy()
    choices_model = np.zeros(len(idx))
    for i in range(len(idx)):
        choices_model[i] = mapping[(model_d1[i], model_d2[i])]

    idx_good = np.where(np.logical_and(np.logical_xor(choices_model <= 1, choices_o <= 1),
                                       in_arms[:, 0, -1] != in_arms[:, 1, -1]))
    idx_c = np.where(np.logical_and(choices_model == choices_c, choices_model != choices_o))[0]
    idx_goodc = np.intersect1d(idx_good, idx_c)

    n_bins = 10
    lim1 = [-3, 3]
    idx_table = np.ndarray((n_bins, 2), dtype=object)
    # find index of trials with different evidence for the first choice
    n_bins2 = 10
    lim2 = [0, 3]
    idx_table2 = np.ndarray((n_bins, 2, n_bins2), dtype=object)
    for i in range(len(idx)):
        tm1 = in_time[i, 0, -1]
        tm2 = in_time[i, 1, -1]
        diff_pri = np.abs(in_arms[i, 0, 0] - tm1) - np.abs(in_arms[i, 1, 0] - tm1)
        checker1 = int((diff_pri - lim1[0]) / ((lim1[1] - lim1[0]) / n_bins))
        idx_tm2 = np.argmax(in_time[i, 1, :]) - 1

        key_i = np.argmax(keys[i, :, idx_tm2])
        if checker1 == n_bins:
            checker1 -= 1

        if in_arms[i, 0, 0] < in_arms[i, 1, 0]:
            lr = 0
        else:
            lr = 1

        if idx_table[checker1, lr] is None:
            idx_table[checker1, lr] = [i]
        else:
            idx_table[checker1, lr].append(i)

        diff_sec = min(abs(tm2 - in_arms[i, mapping1[key_i], 0]))
        checker2 = int((diff_sec - lim2[0]) / ((lim2[1] - lim2[0]) / n_bins2))
        if checker2 >= n_bins2:
            checker2 = n_bins2 - 1

        if idx_table2[checker1, lr, checker2] is None:
            idx_table2[checker1, lr, checker2] = [i]
        else:
            idx_table2[checker1, lr, checker2].append(i)

    colormap1 = 'jet'
    cmap1 = plt.get_cmap(colormap1, n_bins)
    colors1 = cmap1(np.arange(0, cmap1.N))
    colors1 = np.flip(colors1, axis=0)

    colormap2 = 'jet'
    cmap2 = plt.get_cmap(colormap2, n_bins)
    colors2 = cmap2(np.arange(0, cmap2.N))
    colors2 = np.flip(colors2, axis=0)

    fig = plt.figure()

    ax = fig.add_subplot(1, 3, 1, projection='3d')
    lgd = []
    handles = []
    for i in range(n_bins):
        for j in range(2):  # [0,2,7,9]:#
            idx_i = np.array(idx_table[i, j])
            idx_i = np.random.choice(idx_i, min(int(len(idx_i) / 5), 10), replace=False)

            data_i = hs_pc3[idx_i, :, :]
            keys_i = keys[idx_i, :, :]

            idx_tm1 = np.argmax(in_time[idx_i, 0, :], 1) - 1
            idx_tm2 = np.argmax(in_time[idx_i, 1, :], 1) - 1

            for k in range(len(idx_i)):
                h, = ax.plot(data_i[k, :idx_tm2[k] + 1, 0], data_i[k, :idx_tm2[k] + 1, 1],
                             data_i[k, :idx_tm2[k] + 1, 2], color=colors1[i, :], linewidth=2, alpha=0.7)

                if keys_i[k, 0, -1] > 0.5:
                    c = [1, 0, 0]
                else:
                    c = [0, 0, 1]
            ax.scatter(hs_pc3[idx_i, idx_tm1, 0], hs_pc3[idx_i, idx_tm1, 1], hs_pc3[idx_i, idx_tm1, 2],
                       color=[0, 0, 0], s=10, alpha=0.8, marker='<')
            ax.scatter(hs_pc3[idx_i, idx_tm2, 0], hs_pc3[idx_i, idx_tm2, 1], hs_pc3[idx_i, idx_tm2, 2],
                       color=[0, 0, 0], s=10, alpha=0.8, marker='8')
        lgd.append(np.round(lim1[0] + i * ((lim1[1] - lim1[0]) / n_bins), 2))
        handles.append(h)

        # idx_left = keys_i[:, 0, -1] >= 0.5
        # idx_right = keys_i[:, 0, -1] < 0.5
        # ax.scatter(data_i[idx_left, -1, 0], data_i[idx_left, -1, 1], data_i[idx_left, -1, 2], color=c, s=20, alpha=0.8)
        # ax.scatter(data_i[idx_right, -1, 0], data_i[idx_right, -1, 1], data_i[idx_right, -1, 2], color=c, s=20,
        #            alpha=0.8)

    ax.legend(handles, lgd)

    ax = fig.add_subplot(1, 3, 2, projection='3d')
    lgd = []
    handles = []
    for i in range(n_bins):
        for j in range(2):
            idx_i = np.array(idx_table[i, j])
            # idx_i = np.random.choice(idx_i, min(int(len(idx_i) / 5), 10), replace=False)
            idx_tm1 = np.argmax(in_time[idx_i, 0, :], 1) - 1
            idx_tm2 = np.argmax(in_time[idx_i, 1, :], 1) - 1
            x0 = hs_pc3[idx_i, idx_tm2, 0].mean()
            y0 = hs_pc3[idx_i, idx_tm2, 1].mean()
            z0 = hs_pc3[idx_i, idx_tm2, 2].mean()
            ax.scatter(x0, y0, z0, color=colors1[i, :], s=30, alpha=0.8)

            # data_i = hs_pc3[idx_i, :, :]
            # keys_i = keys[idx_i, :, :]
            # idx_left = keys_i[:, 0, -1] >= 0.5
            # idx_right = keys_i[:, 0, -1] < 0.5
            # ax.scatter(data_i[idx_left, -1, 0], data_i[idx_left, -1, 1], data_i[idx_left, -1, 2], color=[1,0,0], s=20, alpha=0.8)
            # ax.scatter(data_i[idx_right, -1, 0], data_i[idx_right, -1, 1], data_i[idx_right, -1, 2], color=[0,0,1], s=20,
            #            alpha=0.8)

            for k in range(n_bins2):
                idx_k = idx_table2[i, j, k]
                if idx_k is not None:
                    x1 = hs_pc3[idx_k, -1, 0].mean()
                    y1 = hs_pc3[idx_k, -1, 1].mean()
                    z1 = hs_pc3[idx_k, -1, 2].mean()
                    h = ax.plot([x0, x1], [y0, y1], [z0, z1], color=colors2[k, :], alpha=0.8)

                    lgd.append(np.round(lim2[0] + i * ((lim2[1] - lim2[0]) / n_bins2), 2))
                    handles.append(h)

    ax.legend(handles, lgd)

    ax = fig.add_subplot(1, 3, 3, projection='3d')
    lgd = []
    handles = []
    for i in range(n_bins):
        for j in range(2):  # [0,2,7,9]:#
            idx_i = np.array(idx_table[i, j])
            idx_i = np.random.choice(idx_i, min(int(len(idx_i) / 5), 10), replace=False)

            data_i = hs_pc3[idx_i, :, :]
            keys_i = keys[idx_i, :, :]

            idx_tm1 = np.argmax(in_time[idx_i, 0, :], 1) - 1
            idx_tm2 = np.argmax(in_time[idx_i, 1, :], 1) - 1

            for k in range(len(idx_i)):
                h, = ax.plot(data_i[k, :, 0], data_i[k, :, 1],
                             data_i[k, :, 2], color=colors1[i, :], linewidth=2, alpha=0.5)

            idx_left = keys_i[:, 0, -1] >= 0.5
            idx_right = keys_i[:, 0, -1] < 0.5
            ax.scatter(data_i[idx_left, -1, 0], data_i[idx_left, -1, 1], data_i[idx_left, -1, 2], color=[1, 0, 0], s=20,
                       alpha=0.6)
            ax.scatter(data_i[idx_right, -1, 0], data_i[idx_right, -1, 1], data_i[idx_right, -1, 2], color=[0, 0, 1],
                       s=20,
                       alpha=0.6)

            # ax.scatter(hs_pc3[idx_i, idx_tm1, 0], hs_pc3[idx_i, idx_tm1, 1], hs_pc3[idx_i, idx_tm1, 2],
            #            color=[0, 0, 0], s=10, alpha=0.8, marker='<')
            # ax.scatter(hs_pc3[idx_i, idx_tm2, 0], hs_pc3[idx_i, idx_tm2, 1], hs_pc3[idx_i, idx_tm2, 2],
            #            color=[0, 0, 0], s=10, alpha=0.8, marker='8')
        lgd.append(np.round(lim1[0] + i * ((lim1[1] - lim1[0]) / n_bins), 2))
        handles.append(h)

        # idx_left = keys_i[:, 0, -1] >= 0.5
        # idx_right = keys_i[:, 0, -1] < 0.5
        # ax.scatter(data_i[idx_left, -1, 0], data_i[idx_left, -1, 1], data_i[idx_left, -1, 2], color=c, s=20, alpha=0.8)
        # ax.scatter(data_i[idx_right, -1, 0], data_i[idx_right, -1, 1], data_i[idx_right, -1, 2], color=c, s=20,
        #            alpha=0.8)

    ax.legend(handles, lgd)
    plt.show()
