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
    """
    Analyzes psychometric performance of the model by computing accuracy across different difficulty levels.
    
    Args:
        params: Dictionary containing model parameters and data generators
    """
    def _acc_ext_supervised(model_results):
        """Helper function to get model choices for supervised learning"""
        outputs = model_results['outputs'].detach().numpy()
        model_d1 = np.argmax(np.mean(outputs[:, :2, 25:], 2), 1)
        model_d2 = np.argmax(np.mean(outputs[:, 2:, 25:], 2), 1)
        choices_model = np.zeros(len(n_trials))
        for j in range(len(n_trials)):
            choices_model[j] = mapping_label[(model_d1[j], model_d2[j])]
        return choices_model

    def _acc_func(model_results):
        """Helper function to get model choices based on key values"""
        keys_batch = model_results['keys'][:, :, 25:]

        # Get binary choices based on mean key values
        model_d1 = torch.mean(keys_batch[:, 0, :], 1) < 0.5
        model_d2 = torch.mean(keys_batch[:, 1, :], 1) < 0.5

        # Convert to combined choice labels
        choices_model = np.zeros(n_trials)
        for j in range(n_trials):
            choices_model[j] = mapping_label[(int(model_d1[j]), int(model_d2[j]))]

        return choices_model

    # Extract parameters
    save_path = params['save_path']
    net = params['net']
    task_data_generator = params['task_data_generator']
    label_func = eval(params['trainer_params']['acc_func'])

    # Get trial data
    trials = task_data_generator()
    in_arms = trials['in_arms']
    in_time = trials['in_time']
    labels = trials['labels'][:, :, -1]
    n_trials = in_arms.shape[0]

    # Define mapping from binary choices to combined labels
    mapping_label = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    # Convert labels to combined format
    labels_d1 = np.argmax(labels[:, :2], 1)
    labels_d2 = np.argmax(labels[:, 2:], 1)
    labels_comb = np.zeros(n_trials)
    for i in range(n_trials):
        labels_comb[i] = mapping_label[(labels_d1[i], labels_d2[i])]

    # Get model predictions
    inputs = np.concatenate((in_arms, in_time), axis=1)
    inputs = torch.from_numpy(inputs)
    model_results = net(copy.deepcopy(inputs))
    choices_model = label_func(model_results)

    # Initialize arrays for storing accuracies
    acc_each = np.zeros((4, 4))  # Accuracy for each difficulty combination
    acc_margin0 = np.zeros(4)    # Marginal accuracy for horizontal choice
    acc_margin1 = np.zeros(4)    # Marginal accuracy for vertical choice
    idx_each = np.ndarray((4, 4), dtype=object)      # Trial indices for each difficulty combo
    idx_margin0 = np.ndarray((4), dtype=object)      # Trial indices for horizontal margins
    idx_margin1 = np.ndarray((4), dtype=object)      # Trial indices for vertical margins

    # Map difficulty levels to indices
    mapping_idx = {
        3: 0,
        2: 1,
        1: 2,
        0: 3
    }

    # Group trials by difficulty levels
    for i in range(n_trials):
        # Get horizontal difficulty index
        idx0 = mapping_idx[np.abs(in_arms[i, 0, -1] - in_arms[i, 1, -1])]

        # Get vertical difficulty index based on chosen path
        if labels_d1[i] == 0:
            idx1 = mapping_idx[np.abs(in_arms[i, 2, -1] - in_arms[i, 3, -1])]
        else:
            idx1 = mapping_idx[np.abs(in_arms[i, 4, -1] - in_arms[i, 5, -1])]

        # Store trial indices for each difficulty combination
        if idx_each[idx0, idx1] is None:
            idx_each[idx0, idx1] = []
        else:
            idx_each[idx0, idx1].append(i)

        # Store trial indices for marginal difficulties
        if idx_margin0[idx0] is None:
            idx_margin0[idx0] = []
        else:
            idx_margin0[idx0].append(i)

        if idx_margin1[idx1] is None:
            idx_margin1[idx1] = []
        else:
            idx_margin1[idx1].append(i)

    # Calculate accuracies for each difficulty level
    for i in range(4):
        # Calculate marginal accuracies
        acc_margin0[i] = sum(choices_model[idx_margin0[i]] == labels_comb[idx_margin0[i]]) / len(idx_margin0[i])
        acc_margin1[i] = sum(choices_model[idx_margin1[i]] == labels_comb[idx_margin1[i]]) / len(idx_margin1[i])
        
        # Calculate accuracies for each difficulty combination
        for j in range(4):
            idx_ij = idx_each[i, j]
            acc_each[i, j] = sum(choices_model[idx_ij] == labels_comb[idx_ij]) / len(idx_ij)

    # Plot results
    fig, ax = plt.subplots(1, 3, figsize=(20, 3))
    cmap = 'jet'
    
    # Plot accuracy heatmap
    h = ax[0].imshow(np.flip(acc_each, axis=1), cmap=mpl.colormaps[cmap], vmin=0, vmax=1)
    ax[0].set_xlabel('Horizontal Diff.')
    ax[0].set_ylabel('Vertical Diff.')
    
    # Plot marginal accuracies
    ax[1].imshow(np.flip(acc_margin0[:, np.newaxis], axis=0), cmap=mpl.colormaps[cmap], vmin=0, vmax=1)
    ax[1].set_title('Horizontal marginal')
    ax[2].imshow(np.flip(acc_margin1[:, np.newaxis], axis=0), cmap=mpl.colormaps[cmap], vmin=0, vmax=1)
    ax[2].set_title('Vertical marginal')
    
    fig.colorbar(h, ax=ax[0])
    plt.show()


def saccade(params):
    """
    Visualizes saccade trajectories for different trial conditions.
    
    Args:
        params: Dictionary containing:
            task_data_generator: Generator for task trial data
            net: Neural network model
    """
    # Extract parameters
    task_data_generator = params['task_data_generator']
    net = params['net']
    trials = task_data_generator()
    in_arms = trials['in_arms']
    tm = trials['in_time']
    idx = 0

    # Define example trials with different conditions
    trial1 = [4, 7, 5, 7, 6, 4, 4, 7]  # easy/easy
    trial2 = [4, 6, 6, 7, 5, 4, 6, 7]  # hierarchical
    trial3 = [6, 5, 7, 5, 4, 4, 5, 7]  # counterfactual
    trial_list = [trial1, trial2, trial3]

    # Create figure with 3 subplots
    fig, ax = plt.subplots(1, 3, figsize=(10, 3))

    # Process each trial
    for t in range(len(trial_list)):
        np.random.seed(3)

        trial = trial_list[t]
        mannual = True
        if mannual:
            # Set arm lengths and time markers
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
                # Process time markers
                tm1_ = int(np.round(tm1))
                tm2_ = int(np.round(tm2))
                tm1_ = max(tm1_, 1)
                tm2_ = max(tm2_, 1)
                
                # Set time marker values with linear ramps
                tm[idx, 0, 5:5 + tm1_ + 1] = np.arange(tm1_ + 1) / tm1_ * tm1
                tm[idx, 0, 5 + tm1_:] = tm1

                tm[idx, 1, 5 + tm1_:5 + tm1_ + tm2_ + 1] = np.arange(tm2_ + 1) / tm2_ * tm2
                tm[idx, 1, 5 + tm1_ + tm2_:] = tm2

        # Get model predictions
        inputs = np.concatenate((in_arms, tm), axis=1)
        results = net(torch.from_numpy(np.expand_dims(inputs[idx, :, :], 0)))
        keys = torch.squeeze(results['keys'])
        keys = keys.detach().numpy()
        prs = torch.squeeze(results['prs'])
        prs = prs.detach().numpy()

        tm_real = np.squeeze(results['tm_modified'])

        # Define time epochs for analysis
        pre = np.arange(4)
        epoch1 = np.arange(6 + tm1_, 6 + tm1_ + 2)  # 4 time points after tm1
        epoch2 = np.arange(tm.shape[2] - 4, tm.shape[2] - 1)  # last 4 time points

        # Calculate choice points based on key values
        choice1 = np.array([1 - np.mean(keys[0, epoch1]), 0])
        choice2 = np.array([1 - np.mean(keys[0, epoch2]), 0.1])
        choice3 = np.array([1 - np.mean(keys[0, epoch2]), np.mean(keys[1, epoch2])])

        # Define trajectory points
        point0 = [0, 0]  # Start point
        point1 = [(1 - choice1[0]) * trial[0] * -1 + choice1[0] * trial[1], 0]  # First decision point
        point4 = [(1 - choice3[0]) * trial[0] * -1 + choice3[0] * trial[1],  # Final decision point
                  (1 - choice3[0]) * (choice3[1] * trial[2] + (1 - choice3[1]) * trial[3] * -1) + choice3[0] * (
                          choice3[1] * trial[4] + (1 - choice3[1]) * trial[5] * -1)]
        points = np.array([point0, point1, point4])

        # Plot maze structure and trajectories
        lgd = str(trial)
        ax1 = ax[t]
        width = 10
        color = (.4, .4, .4, .3)
        
        # Draw maze structure
        ax1.plot([0, 0], [0, 5], linewidth=width, color=(.7, .7, .7, .3))  # Vertical center line
        ax1.plot([0, -trial[0]], [0, 0], linewidth=width, color=color)  # Left horizontal arm
        ax1.plot([0, trial[1]], [0, 0], linewidth=width, color=color)  # Right horizontal arm
        ax1.plot([-trial[0], -trial[0]], [0, trial[2]], linewidth=width, color=color)  # Left upper vertical
        ax1.plot([-trial[0], -trial[0]], [0, -trial[3]], linewidth=width, color=color)  # Left lower vertical
        ax1.plot([trial[1], trial[1]], [0, trial[4]], linewidth=width, color=color)  # Right upper vertical
        ax1.plot([trial[1], trial[1]], [0, -trial[5]], linewidth=width, color=color)  # Right lower vertical

        # Plot start point and trajectory arrows
        ax1.scatter(0, 0, s=200, color=(.7, 0, 0, .7), label=lgd)
        for i in range(len(points) - 1):
            ax1.quiver(points[i, 0], points[i, 1], points[i + 1, 0] - points[i, 0], points[i + 1, 1] - points[i, 1],
                       angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=10, headlength=9,
                       color=(1, 0, 0, 0.5))

        # Draw time marker indicators
        left_end = -8
        ax1.quiver(left_end, -8, 16, 0,
                   angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=10, headlength=9,
                   color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1, left_end + 1], [-8, -7], linewidth=3, color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1 + trial[6], left_end + 1 + trial[6]], [-8, -7], linewidth=3, color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1 + trial[6] + trial[7], left_end + 1 + trial[6] + trial[7]], [-8, -7], linewidth=3,
                 color=(1, 0, 0, 0.5))

        # Set plot properties
        ax1.legend()
        ax1.set_xlim([-10, 10])
        ax1.set_ylim([-10, 10])

    plt.show()


def viz(params):
    """
    Visualizes model behavior by plotting arm lengths, keys, and saccade trajectories.
    
    Args:
        params: Dictionary containing:
            task_data_generator: Generator for task trial data
            net: Neural network model
    """
    # Extract parameters
    task_data_generator = params['task_data_generator']
    net = params['net']
    trials = task_data_generator()
    in_arms = trials['in_arms']
    tm = trials['in_time']
    idx = 0

    # Define example trials
    trial1 = [4, 7, 5, 7, 6, 4, 4, 7]  # easy/easy
    trial2 = [4, 6, 6, 7, 5, 4, 6, 7]  # hierarchical  
    trial3 = [5, 4, 7, 6, 4, 4, 4, 7]  # counterfactual
    # trial2 = [7, 4, 5, 6, 7, 4, 7, 6]  # easy/hard
    # trial3 = [5, 6, 7, 4, 4, 7, 6, 7]  # hard/easy
    # trial4 = [5, 6, 4, 5, 7, 6, 6, 7]  # hard/hard
    trial_list = [trial1, trial2, trial3]

    # Process each trial
    for t in range(len(trial_list)):
        trial = trial_list[t]
        mannual = True
        if mannual:
            # Set arm lengths
            in_arms[idx, 0, :] = trial[0]
            in_arms[idx, 1, :] = trial[1]
            in_arms[idx, 2, :] = trial[2]
            in_arms[idx, 3, :] = trial[3]
            in_arms[idx, 4, :] = trial[4]
            in_arms[idx, 5, :] = trial[5]
            
            # Set time markers
            tm[idx, :, :] = 0
            tm1 = trial[6]
            tm2 = trial[7]

            if True:
                # Round time markers and ensure minimum value of 1
                tm1_ = int(np.round(tm1))
                tm2_ = int(np.round(tm2))
                tm1_ = max(tm1_, 1)
                tm2_ = max(tm2_, 1)
                
                # Create ramping time markers
                tm[idx, 0, 5:5 + tm1_ + 1] = np.arange(tm1_ + 1) / tm1_ * tm1
                tm[idx, 0, 5 + tm1_:] = tm1

                tm[idx, 1, 5 + tm1_:5 + tm1_ + tm2_ + 1] = np.arange(tm2_ + 1) / tm2_ * tm2
                tm[idx, 1, 5 + tm1_ + tm2_:] = tm2

        # Prepare inputs and get model predictions
        inputs = np.concatenate((in_arms, tm), axis=1)
        np.random.seed(0)
        results = net(torch.from_numpy(np.expand_dims(inputs[idx, :, :], 0)))
        keys = torch.squeeze(results['keys'])
        keys = keys.detach().numpy()
        prs = torch.squeeze(results['prs'])
        prs = prs.detach().numpy()
        tm_real = np.squeeze(results['tm_modified'])

        # Setup plotting parameters
        labels = trials['labels']
        n_time = in_arms.shape[2]
        n_arms = in_arms.shape[1]
        colormap = 'Set1'
        cmap = plt.get_cmap(colormap, n_arms)
        colors = cmap(np.arange(0, cmap.N))

        # Create figure with 3 subplots
        fig, ax = plt.subplots(3, 1)
        x = np.arange(n_time)

        # Plot primary arm lengths
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

        # Plot secondary arm lengths
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

        # Plot keys
        keys = keys[:, 1:]
        x = x[1:]
        handles = []
        for i in range(2):
            h_i, = ax[2].plot(x, keys[i, :], color=colors[i, :])
            handles.append(h_i)
        ax[2].set_ylabel('Keys')
        lgd = ('Horizontal', 'Vertical')
        ax[2].legend(handles, lgd)

        # Calculate saccade trajectories
        pre = np.arange(4)
        epoch1 = np.arange(6 + tm1_, 6 + tm1_ + 2)  # 4 time points after tm1 presentation
        epoch2 = np.arange(tm.shape[2] - 4, tm.shape[2] - 1)  # last 4 time points

        # Calculate choice points based on key values
        choice1 = np.array([1 - np.mean(keys[0, epoch1]), 0])
        choice2 = np.array([1 - np.mean(keys[0, epoch2]), 0.1])
        choice3 = np.array([1 - np.mean(keys[0, epoch2]), np.mean(keys[1, epoch2])])
        y1 = 0.2
        y2 = -0.2

        # Define trajectory points
        point0 = [0, 0]
        point1 = [(1 - choice1[0]) * trial[0] * -1 + choice1[0] * trial[1], 0]
        point4 = [(1 - choice3[0]) * trial[0] * -1 + choice3[0] * trial[1],
                  (1 - choice3[0]) * (choice3[1] * trial[2] + (1 - choice3[1]) * trial[3] * -1) + choice3[0] * (
                          choice3[1] * trial[4] + (1 - choice3[1]) * trial[5] * -1)]
        points = np.array([point0, point1, point4])

        # Plot saccade trajectories
        epoch_colors = [[.5, .5, .5], colors[0, :3], colors[1, :3], colors[3, :3]]
        handles = []
        lgd = str(trial)
        fig1, ax1 = plt.subplots()
        
        # Draw maze structure
        width = 10
        color = (.4, .4, .4, .3)
        ax1.plot([0, 0], [0, 5], linewidth=width, color=(.7, .7, .7, .3))
        ax1.plot([0, -trial[0]], [0, 0], linewidth=width, color=color)
        ax1.plot([0, trial[1]], [0, 0], linewidth=width, color=color)
        ax1.plot([-trial[0], -trial[0]], [0, trial[2]], linewidth=width, color=color)
        ax1.plot([-trial[0], -trial[0]], [0, -trial[3]], linewidth=width, color=color)
        ax1.plot([trial[1], trial[1]], [0, trial[4]], linewidth=width, color=color)
        ax1.plot([trial[1], trial[1]], [0, -trial[5]], linewidth=width, color=color)

        # Plot start point and trajectory arrows
        ax1.scatter(0, 0, s=200, color=(.7, 0, 0, .7), label=lgd)
        for i in range(len(points) - 1):
            ax1.quiver(points[i, 0], points[i, 1], points[i + 1, 0] - points[i, 0], points[i + 1, 1] - points[i, 1],
                       angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=10, headlength=9,
                       color=(1, 0, 0, 0.5))

        # Draw time marker indicators
        left_end = -8
        ax1.quiver(left_end, -8, 16, 0,
                   angles='xy', scale_units='xy', scale=1, width=0.005, headwidth=10, headlength=9,
                   color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1, left_end + 1], [-8, -7], linewidth=3, color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1 + trial[6], left_end + 1 + trial[6]], [-8, -7], linewidth=3, color=(1, 0, 0, 0.5))
        ax1.plot([left_end + 1 + trial[6] + trial[7], left_end + 1 + trial[6] + trial[7]], [-8, -7], linewidth=3,
                 color=(1, 0, 0, 0.5))

        # Set plot properties
        ax1.legend()
        ax1.set_xlim([-10, 10])
        ax1.set_ylim([-10, 10])
        plt.show()



def KL_divergence(P, Q):
    """
    Calculates the Kullback-Leibler divergence between two probability distributions.
    
    Args:
        P: Array of shape (n_bootstrap, n_samples, n_classes) containing true distributions
        Q: Array of shape (n_bootstrap, n_samples, n_classes) containing model distributions
        
    Returns:
        Array of shape (n_bootstrap) containing mean KL divergence for each bootstrap sample
        
    Notes:
        - Small epsilon (1e-2) is added to probabilities to avoid numerical issues
        - Probabilities are renormalized after adding epsilon
        - For each bootstrap sample, returns mean KL divergence across all samples
    """
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
    """
    Analyzes how well the model matches different strategies by comparing negative log likelihoods.
    
    Args:
        params: Dictionary containing:
            save_path: Path to save results
            net: Neural network model
            task_data_generator: Generator for task trial data
            trainer_params: Training parameters including accuracy function
            n_bootstrap: Number of bootstrap iterations
    """

    def _acc_ext_supervised(model_results):
        """Convert model outputs to choice labels for supervised learning"""
        outputs = model_results['outputs'].detach().numpy()
        model_d1 = np.argmax(np.mean(outputs[:, :2, 25:], 2), 1)
        model_d2 = np.argmax(np.mean(outputs[:, 2:, 25:], 2), 1)
        n_trials = model_results['keys'].shape[0]
        choices_model = np.zeros(n_trials)
        for j in range(n_trials):
            choices_model[j] = mapping[(model_d1[j], model_d2[j])]
        return choices_model
    
    
    def _acc_func(model_results):
        """Convert model key activations to choice labels"""
        keys_batch = model_results['keys'][:, :, 25:]

        d1 = torch.mean(keys_batch[:, 0, :], 1) < 0.5
        d2 = torch.mean(keys_batch[:, 1, :], 1) < 0.5
        n_trials = model_results['keys'].shape[0]

        choices_model = np.zeros(n_trials)
        for j in range(n_trials):
            choices_model[j] = mapping[(int(d1[j]), int(d2[j]))]

        return choices_model

    # Extract parameters
    save_path = params['save_path']
    net = params['net']
    task_data_generator = params['task_data_generator']
    label_func = eval(params['trainer_params']['acc_func'])
    n_bootstrap = params['n_bootstrap']
    
    # Define mapping from binary decisions to choice labels
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }
    wb = task_data_generator.wb
    wb_inc = task_data_generator.wb_inc
    
    # Parameters for threshold search
    thresh_range = [0, 1]
    n_search = 5

    # Load or compute counterfactual model emissions
    emission_c_file = save_path.parents[0] / (str(np.around(wb_inc, decimals=3))+'.ems')
    if emission_c_file.exists():
        results = torch.load(emission_c_file)
        emission_c_dict = results['emission_c']
    else:
        eps=1e-3

        def _loss_func(trials, wb, wb_inc, threshold):
            """Compute negative log likelihood loss for threshold selection"""
            emission = p_models.counterfactual_emission(trials, wb, wb_inc, threshold=threshold, n_sim=1000)
            labels = trials['labels']
            result = -np.mean(np.log(emission[np.arange(len(emission)), labels.astype(int)] + eps))
            return result
            
        # Generate trial data
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

        # Binary search for optimal threshold
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

        # Use optimal threshold to compute emissions
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

        # Save counterfactual emissions
        results = {
            'emission_c': emission_c_dict
        }
        torch.save(results, emission_c_file)

    # Load or compute emissions for other models
    emission_file = save_path.parents[1] / 'emission.dict'
    if emission_file.exists():
        results = torch.load(emission_file)
        emission_o_dict = results['emission_o']
        emission_p_dict = results['emission_p']
        emission_h_dict = results['emission_h']
    else:
        # Generate trial data
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
        
        # Compute emissions for each model
        emission_o = p_models.optimal_emission(trials, wb=0.15, n_sim=10000)
        emission_p = p_models.postdictive_emission(trials, wb=0.15, n_sim=10000)
        emission_h = p_models.hierarchy_emission(trials, wb=0.15, n_sim=10000)

        # Store emissions in dictionaries
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

        # Save all emissions
        results = {
            'emission_o': emission_o_dict,
            'emission_p': emission_p_dict,
            'emission_h': emission_h_dict
        }

        torch.save(results, 'emission.dict')
    
    # Initialize array for negative log likelihoods
    metric_all = np.zeros((n_bootstrap, 4))  # o, h, p, c

    # Bootstrap evaluation
    for it in range(n_bootstrap):
        trials = task_data_generator(seed=it)
        idx_val = np.arange(len(trials['in_arms']))
        in_arms_val = trials['in_arms'][idx_val, :, :]
        in_time_val = trials['in_time'][idx_val, :, :]
        labels_val = trials['labels'][idx_val, :, -1]

        # Convert labels to combined format
        labels_val_d1 = np.argmax(labels_val[:, :2], 1)
        labels_val_d2 = np.argmax(labels_val[:, 2:], 1)
        labels_val_comb = np.zeros(len(idx_val))
        for i in range(len(idx_val)):
            labels_val_comb[i] = mapping[(labels_val_d1[i], labels_val_d2[i])]

        # Create maze tuples for lookup
        mazes = []
        for i in range(in_arms_val.shape[0]):
            mazes.append(tuple(in_arms_val[i, :, -1].astype(int).tolist() + [int(labels_val_comb[i])]))

        # Get model predictions
        inputs = np.concatenate((in_arms_val, in_time_val), axis=1)
        inputs = torch.from_numpy(inputs)
        model_results = net(copy.deepcopy(inputs))
        choices_model = label_func(model_results)

        # Compute negative log likelihoods
        eps=1e-2
        for i in range(len(mazes)):
            maze = tuple(mazes[i])
            choice_rnn = int(choices_model[i])

            metric_all[it, 0] -= np.log(emission_o_dict[maze][choice_rnn] + eps)
            metric_all[it, 1] -= np.log(emission_h_dict[maze][choice_rnn] + eps)
            metric_all[it, 2] -= np.log(emission_p_dict[maze][choice_rnn] + eps)
            metric_all[it, 3] -= np.log(emission_c_dict[maze][choice_rnn] + eps)

        metric_all[it, :] /= len(mazes)

    # Save results
    results = {
        'nll': metric_all
    }

    torch.save(results, str(save_path) + '.nll')

    return results



def AnalysisWMNoiseSweep(params):
    """
    Analyzes working memory noise by comparing model predictions to different strategies.
    
    Args:
        params: Dictionary containing model parameters and data generator
    """
    def perf_label(model_results):
        """Converts model outputs to choice labels"""
        outputs = model_results['outputs'].detach().numpy()
        model_d1 = np.argmax(np.mean(outputs[:, :2, 25:], 2), 1)
        model_d2 = np.argmax(np.mean(outputs[:, 2:, 25:], 2), 1)
        choices_model = np.zeros(len(idx_val))
        for j in range(len(idx_val)):
            choices_model[j] = mapping[(model_d1[j], model_d2[j])]
        return choices_model

    def _acc_func(model_results):
        """Converts model key activations to choice labels"""
        keys_batch = model_results['keys'][:, :, 25:]

        d1 = torch.mean(keys_batch[:, 0, :], 1) < 0.5
        d2 = torch.mean(keys_batch[:, 1, :], 1) < 0.5
        n_trials = model_results['keys'].shape[0]

        choices_model = np.zeros(n_trials)
        for j in range(n_trials):
            choices_model[j] = mapping[(int(d1[j]), int(d2[j]))]

        return choices_model

    # Extract parameters
    save_path = params['save_path']
    net = params['net']
    task_data_generator = params['task_data_generator']
    label_func = eval(params['trainer_params']['acc_func'])
    wb = task_data_generator.wb
    wb_inc = task_data_generator.wb_inc
    iteration = 10

    # Find optimal threshold for counterfactual model on training data
    thresh = np.arange(0.01, 0.2, 0.01)
    trials = task_data_generator()
    
    # Get training data
    idx_train = trials['idx_train']
    in_arms_train = trials['in_arms'][idx_train, :, :]
    in_time_train = trials['in_time'][idx_train, :, :]
    labels_train = trials['labels'][idx_train, :, -1]
    trials_train = {
        'in_arms': in_arms_train,
        'in_time': in_time_train,
        'labels': labels_train
    }

    # Define mapping from binary decisions to choice labels
    mapping = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    # Convert training labels to choice labels
    labels_train_d1 = np.argmax(labels_train[:, :2], 1)
    labels_train_d2 = np.argmax(labels_train[:, 2:], 1)
    labels_train_comb = np.zeros(len(idx_train))
    for i in range(len(idx_train)):
        labels_train_comb[i] = mapping[(labels_train_d1[i], labels_train_d2[i])]

    # Test different thresholds for counterfactual model
    choices_c = np.ndarray(len(thresh), dtype=object)
    acc_c = np.zeros(len(thresh))

    for i, th in enumerate(thresh):
        choices_c[i] = p_models.counterfactual(trials_train, wb, wb_inc, threshold=th)
        acc_c[i] = sum(labels_train_comb == choices_c[i]) / len(choices_c[i])

    # Select threshold with best accuracy
    thresh_c = thresh[np.argmax(acc_c)]

    # Initialize arrays for storing distances between model and strategies
    euc_all = np.zeros((iteration, 4))  # optimal, hierarchical, postdictive, counterfactual

    # Run multiple iterations
    for it in range(iteration):
        # Get validation data
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

        # Convert validation labels to choice labels
        labels_val_d1 = np.argmax(labels_val[:, :2], 1)
        labels_val_d2 = np.argmax(labels_val[:, 2:], 1)
        labels_val_comb = np.zeros(len(idx_val))
        for i in range(len(idx_val)):
            labels_val_comb[i] = mapping[(labels_val_d1[i], labels_val_d2[i])]

        # Get predictions from different strategies
        choices_p = p_models.postdictive(trials_val, wb)
        choices_o = p_models.optimal(trials_val, wb)
        choices_h = p_models.hierarchy(trials_val, wb)
        choices_c = p_models.counterfactual(trials_val, wb, wb_inc, threshold=thresh_c)

        # Get model predictions
        inputs = np.concatenate((in_arms_val, in_time_val), axis=1)
        inputs = torch.from_numpy(inputs)
        model_results = net(copy.deepcopy(inputs))
        choices_model = label_func(model_results)

        # Calculate accuracy of counterfactual model
        acc_c = sum(labels_val_comb == choices_c) / len(choices_c)

        # Setup for coordinate metric calculation
        n_length = len(np.unique(in_arms_val[:, 0, 0]))
        n_diffs = n_length * 2 - 1

        # Initialize arrays for storing choice vectors
        indices = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)
        vector_o = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)
        vector_h = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)
        vector_p = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)
        vector_c = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)
        vector_model = np.ndarray((n_diffs, n_diffs, n_diffs), dtype=object)

        # Define mappings for arm transformations
        mapping10 = {0: 1, 1: 0, 2: 3, 3: 2}  # Map to other vertical arm
        mapping0 = {0: [2, 3], 1: [2, 3], 2: [0, 1], 3: [0, 1]}  # Map to other maze side

        # Calculate choice vectors for each maze condition
        for i in range(len(in_arms_val)):
            arms_i = in_arms_val[i, :, 0]
            tm2_i = in_time_val[i, 1, -1]
            
            # Calculate differences between arms
            diff1 = int(arms_i[0] - arms_i[1] + n_length - 1)
            diff2 = int(arms_i[2] - arms_i[3] + n_length - 1)
            diff3 = int(arms_i[4] - arms_i[5] + n_length - 1)

            # Determine arm ordering
            label_i = labels_val_comb[i]
            order = [label_i, mapping10[label_i]]  # 11, 10 arm

            # Find incorrect primary arms
            arms0 = np.array(mapping0[label_i])
            error0 = np.abs(arms_i[arms0 + 2] - tm2_i)
            arm01_idx = np.random.choice(np.where(error0 == np.min(error0))[0], 1)
            arm01 = arms0[arm01_idx][0]
            arm00 = mapping10[arm01]
            order.append(arm01)
            order.append(arm00)
            order = np.array(order)

            # Store choice vectors for each strategy
            for vector, choice in [(vector_o, choices_o), (vector_h, choices_h),
                                 (vector_p, choices_p), (vector_c, choices_c),
                                 (vector_model, choices_model)]:
                if vector[diff1, diff2, diff3] is None:
                    vector[diff1, diff2, diff3] = [np.where(order == choice[i])[0][0]]
                else:
                    vector[diff1, diff2, diff3].append(np.where(order == choice[i])[0][0])

        # Flatten vectors for comparison
        vector_o = vector_o.flatten()
        vector_h = vector_h.flatten()
        vector_p = vector_p.flatten()
        vector_c = vector_c.flatten()
        vector_model = vector_model.flatten()

        # Calculate Euclidean distances between model and strategies
        euc_o, euc_h, euc_p, euc_c = [], [], [], []
        
        for i in range(len(vector_model)):
            if vector_model[i] is not None:
                # Convert choices to one-hot vectors
                v_m = np.array(vector_model[i])
                v_m1 = np.zeros((len(v_m), 4))
                v_m1[np.arange(v_m1.shape[0]), v_m] = 1
                v_m1 = np.mean(v_m1, axis=0)

                for vec, choices in [(euc_o, vector_o), (euc_h, vector_h),
                                   (euc_p, vector_p), (euc_c, vector_c)]:
                    v = np.array(choices[i])
                    v1 = np.zeros((len(v), 4))
                    v1[np.arange(v1.shape[0]), v] = 1
                    v1 = np.mean(v1, axis=0)
                    vec.append(v1 - v_m1)

        # Store mean Euclidean distances
        euc_all[it, 0] = np.mean(la.norm(np.array(euc_o)[:, :], ord=2, axis=1))
        euc_all[it, 1] = np.mean(la.norm(np.array(euc_h)[:, :], ord=2, axis=1))
        euc_all[it, 2] = np.mean(la.norm(np.array(euc_p)[:, :], ord=2, axis=1))
        euc_all[it, 3] = np.mean(la.norm(np.array(euc_c)[:, :], ord=2, axis=1))

        print(it)

    # Save results
    results = {
        'euc_dist': euc_all,
    }
    torch.save(results, str(save_path) + '.euc')

    return results


def SwitchFrequency(params):
    """
    Calculates the frequency of decision switches for different input differences.
    
    Args:
        params: Dictionary containing:
            save_path: Path to save results
            net: Neural network model
            task_data_generator: Generator for task trial data
            
    Returns:
        Dictionary containing switch frequencies for each input difference level
    """
    save_path = params['save_path']
    net = params['net']
    task_data_generator = params['task_data_generator']

    # Initialize array to store frequencies for 4 different input differences
    freq = np.zeros(4)  

    # Get trial data
    trials = task_data_generator()
    in_arms = trials['in_arms']  # Input arm values
    in_time = trials['in_time']  # Time indicators
    labels = trials['labels'][:, :, -1]  # Trial labels

    # Prepare inputs for network
    inputs = np.concatenate((in_arms, in_time), axis=1)
    inputs = torch.from_numpy(inputs)
    
    # Get model predictions
    model_results = net(copy.deepcopy(inputs))
    keys = model_results['keys'].detach().numpy()
    
    # Find time marker positions
    idx_tm = np.argmax(in_time, axis=2)
    n_trials = inputs.shape[0]
    
    # Count switches for each trial
    switch_counts = np.zeros(n_trials)
    for n in range(n_trials):
        # Get binary decisions after time marker
        switches = keys[n, 0, idx_tm[n, 0]:] < 0.5
        # Count number of switches using XOR between adjacent timepoints
        switch_counts[n] = np.sum(np.logical_xor(np.roll(switches, 1), switches)[1:])

    # Calculate mean switch frequency for each input difference level
    diff_h = abs(inputs[:, 0, 0] - inputs[:, 1, 0]).detach().numpy()
    for diff in range(4):
        freq[diff] = switch_counts[diff_h == diff].mean()

    # Store and save results
    results = {
        'freq': freq,
    }

    torch.save(results, str(save_path) + '.sf')
    return results
