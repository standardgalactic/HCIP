import time
import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from os.path import exists
import os, psutil
import utils.writeLog as writeLog


def _PlotProgress(params):
    c_loss_train = params['loss_train']
    c_acc_train = params['acc_train']
    c_loss_val = params['loss_val']
    c_acc_val = params['acc_val']
    epoch = params['epoch']
    fig, ax = plt.subplots()
    line1, = ax.plot(np.arange(epoch) + 1, c_acc_train)
    line2, = ax.plot(np.arange(epoch) + 1, c_acc_val)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend([line1, line2], ['training', 'validation'])
    ax.set_ylim(0, 1)
    ax.set_xlim(1, epoch)
    plt.show()


def _acc_func(results_batch, labels_batch):
    start = 25
    keys_batch = results_batch['keys'][:, :, start:]

    d1 = torch.mean(keys_batch[:, 0, :], 1) < 0.5
    d2 = torch.mean(keys_batch[:, 1, :], 1) < 0.5

    d1_label = torch.argmax(torch.mean(labels_batch[:, :2, start:], 2), 1)
    d2_label = torch.argmax(torch.mean(labels_batch[:, 2:, start:], 2), 1)

    n_cor = sum(torch.logical_and(d1 == d1_label, d2 == d2_label))
    return n_cor


def _loss_self_supervised(inputs_batch, results_batch, labels_batch):
    in_time = inputs_batch[:, -2:, :]
    idx_tm = np.argmax(in_time, axis=2)
    start = idx_tm[:, 1]

    keys_batch = results_batch['keys']
    prs_batch = results_batch['prs']
    mask = torch.zeros((prs_batch.shape[0], prs_batch.shape[2]))

    for i in range(inputs_batch.shape[0]):
        start_i = start[i]
        mask[i, start_i:] = 1
    key_h = keys_batch[:, 0, :]
    key_v = keys_batch[:, 1, :]

    loss_batch = key_h * prs_batch[:, 0, :] + (1 - key_h) * prs_batch[:, 1, :] + \
                 key_v * prs_batch[:, 2, :] + (1 - key_v) * prs_batch[:, 3, :]
    loss_batch = torch.sum(loss_batch * mask)

    return loss_batch


def _loss_ext_supervised(inputs_batch, results_batch, labels_batch):
    in_time = inputs_batch[:, -2:, :]
    idx_tm = np.argmax(in_time, axis=2)
    start = idx_tm[:, 1]

    keys_batch = results_batch['keys']
    # converted labels
    labels_c_batch = torch.zeros((labels_batch.shape[0], 2, labels_batch.shape[2]))
    labels_c_batch[:, 0, :] = labels_batch[:, 0, :]
    labels_c_batch[:, 1, :] = labels_batch[:, 2, :]
    mask = torch.zeros((keys_batch.shape[0], 2, keys_batch.shape[2]))

    for i in range(inputs_batch.shape[0]):
        start_i = start[i]
        mask[i, :, start_i:] = 1

    loss_func = nn.MSELoss(reduction='mean')
    loss_batch = loss_func(mask*keys_batch, mask*labels_c_batch)
    #
    return loss_batch


def _viz(net, trials, idx, mannual):
    in_arms = trials['in_arms']
    tm = trials['in_time']

    if mannual:
        in_arms[idx, 0, :] = 4
        in_arms[idx, 1, :] = 5
        in_arms[idx, 2, :] = 4
        in_arms[idx, 3, :] = 4
        in_arms[idx, 4, :] = 7
        in_arms[idx, 5, :] = 6
        tm[idx, :, :] = 0
        tm1 = 4
        tm2 = 7
        if True:
            tm1_ = int(np.round(tm1))
            tm2_ = int(np.round(tm2))
            tm1_ = max(tm1_, 1)
            tm2_ = max(tm2_, 1)
            tm[idx, 0, 5:5 + tm1_ + 1] = np.arange(tm1_ + 1) / tm1_ * tm1
            tm[idx, 0, 5 + tm1_:] = tm1

            tm[idx, 1, 5 + tm1_:5 + tm1_ + tm2_ + 1] = np.arange(tm2_ + 1) / tm2_ * tm2
            tm[idx, 1, 5 + tm1_ + tm2_:] = tm2
        else:
            tm[idx, 0, int(5 + np.round(tm1)):] = tm1
            tm[idx, 1, int(5 + np.round(tm1) + np.round(tm2)):] = tm2

    softmax = torch.nn.Softmax(dim=1)
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
    colors = cmap(np.arange(0, cmap.N))

    fig, ax = plt.subplots(3, 3)
    x = np.arange(n_time)

    # input primary arm lengths
    handles = []
    lgd = ('Arm 0: ' + str(in_arms[idx, 0, 0]), 'Arm 1: ' + str(in_arms[idx, 1, 0]), 'Tm 1: '
           + str(np.max(tm[idx, 0, :])))
    for i in range(2):
        h_i, = ax[0, 0].plot(x, in_arms[idx, i, :], color=colors[i, :])
        handles.append(h_i)
    h_i, = ax[0, 0].plot(x, tm_real[0, :], linestyle='--')
    handles.append(h_i)
    ax[0, 0].set_ylim([-1, 15])
    ax[0, 0].set_ylabel('Primary arms')
    ax[0, 0].legend(handles, lgd)

    # input secondary arm lengths
    handles = []
    lgd = (
        'Arm 2: ' + str(in_arms[idx, 2, 0]), 'Arm 3: ' + str(in_arms[idx, 3, 0]),
        'Arm 4: ' + str(in_arms[idx, 4, 0]),
        'Arm 5: ' + str(in_arms[idx, 5, 0]), 'Tm2: ' + str(np.max(tm[idx, 1, :])))
    for i in range(2, 6):
        h_i, = ax[1, 0].plot(x, in_arms[idx, i, :], color=colors[i, :])
        handles.append(h_i)
    h_i, = ax[1, 0].plot(x, tm_real[1, :], linestyle='--')
    handles.append(h_i)
    ax[1, 0].set_ylim([-1, 15])
    ax[1, 0].set_ylabel('Secondary arms')
    ax[1, 0].legend(handles, lgd)

    # pr1
    lgd = ('P(Arm 0)', 'P(Arm 1)')
    handles = []
    for i in range(2):
        h_i, = ax[0, 1].plot(x, prs[i, :])
        handles.append(h_i)
    ax[0, 1].set_ylabel('Pr 1')
    # ax[0, 1].set_ylim([0, 1])
    ax[0, 1].legend(handles, lgd)

    # pr2
    lgd = ('P(Top)', 'P(Bottom)')
    handles = []
    for i in range(2, 4):
        h_i, = ax[1, 1].plot(x, prs[i, :])
        handles.append(h_i)
    ax[1, 1].set_ylabel('Pr 2')
    # ax[1, 1].set_ylim([0, 1])
    ax[1, 1].legend(handles, lgd)

    # keys
    handles = []
    for i in range(2):
        h_i, = ax[2, 0].plot(x, keys[i, :], color=colors[i, :])
        handles.append(h_i)
    ax[2, 0].set_ylabel('Keys')
    lgd = ('Horizontal', 'Vertical')
    ax[2, 0].legend(handles, lgd)

    # target
    handles = []
    # lgd = ('Arm 2', 'Arm 3', 'Arm 4', 'Arm 5')
    lgd = ('Left', 'Right', 'Up', 'Down')

    for i in range(4):
        h_i, = ax[2, 1].plot(x, labels[idx, i, :], color=colors[i, :])
        handles.append(h_i)
    ax[2, 1].set_ylim([-.2, 1.2])
    ax[2, 1].set_ylabel('Target')
    ax[2, 1].legend(handles, lgd)
    trial = {
        'arms': inputs[idx, :, :],
        'keys': keys
    }
    # torch.save(trial, 'submission/trial1')
    plt.show()
    a = 1


class rnn_trainer:  # train RNN with numerical input output
    def __init__(self, params):
        # Initialize training parameters from input dictionary
        self.batch_size = params['batch_size']  # Number of samples per batch
        self.sweep_max = params['sweep_max']  # Maximum number of sweeps through data
        self.epoch_max = params['epoch_max']  # Maximum number of training epochs
        self.lr = params['lr']  # Learning rate
        self.optimizer = getattr(optim, params['optimizer'])  # Optimization algorithm
        self.val_size = params['val_size']  # Validation set size
        self.stop_after = params['stop_after']  # Stop if no improvement after this many epochs
        self.task_data_generator = params['task_data_generator']  # Function to generate task data
        self.loss_func = eval(params['loss_func'])  # Loss function
        self.acc_func = eval(params['acc_func'])  # Accuracy function

    def train(self, params):
        # Extract parameters and initialize training
        task_data_generator = params['task_data_generator']
        trials = task_data_generator()  # Generate task trials
        save_path = params['save_path']  # Path to save model
        net = params['net']  # Neural network model
        log = writeLog.writeLog(save_path)  # Initialize logging
        epoch_max = self.epoch_max
        idx_train = trials['idx_train']  # Training set indices
        idx_val = trials['idx_val']  # Validation set indices

        # Load existing model if it exists
        if exists(str(save_path) + '.model'):
            saved_data = torch.load(save_path + '.model', map_location=torch.device('cpu'))
            net = saved_data['net']
            print('Model loaded!')
            log.write('Model loaded!\n')

        net.zero_grad()

        # Initialize tracking variables for training progress
        c_loss_train = []  # Training loss history
        c_acc_train = []   # Training accuracy history
        c_loss_val = []    # Validation loss history
        c_acc_val = []     # Validation accuracy history

        epoch = 0
        tic = time.time()
        acc_val_max = 0  # Best validation accuracy

        while True:
            # Shuffle training and validation indices
            idx_train = np.random.choice(idx_train, len(idx_train), replace=False)
            idx_val = np.random.choice(idx_val, len(idx_val), replace=False)

            # Train on training set and evaluate on validation set
            loss_train, acc_train = self._sweep_data(net, trials, idx_train, training=True)
            loss_val, acc_val = self._sweep_data(net, trials, idx_val, training=False)
            epoch += 1

            # Record training progress
            c_loss_train.append(loss_train.data)
            c_acc_train.append(acc_train)
            c_loss_val.append(loss_val.data)
            c_acc_val.append(acc_val)

            # Log progress
            text = str(epoch) + ' ' + str(loss_train.data) + ' ' + str(acc_train) + ' ' + str(
                loss_val.data) + ' ' + str(acc_val) + '\n'
            print(text)
            log.write(text)
            toc = time.time()
            text = str(toc - tic) + '\n'
            print(text)
            log.write(text)
            tic = time.time()

            # Save model if validation accuracy improves
            if acc_val == max(c_acc_val):
                acc_val_max = acc_val
                save_model = {
                    'net': net,
                    'epoch': epoch,
                    'acc_val': acc_val,
                    'loss_val': loss_val
                }
                torch.save(save_model, str(save_path) + '.model')
                print('Model saved!')
                log.write('Model saved!\n')

            # Early stopping check
            if max(np.array(c_acc_val)[-self.stop_after:]) < acc_val_max:
                break

            # Stop if max epochs reached
            if epoch == epoch_max:
                break

            # Reset model if performance is poor
            if acc_val < 0.5:
                trials = task_data_generator()
                idx_train = trials['idx_train']
                idx_val = trials['idx_val']
                net.W_linear_in.reset_parameters()
                net.W_key.reset_parameters()
                net.W_rec.reset_parameters()
                net.zero_grad()
                text = 're-initialize model' + '\n'
                acc_val_max = 0
                c_acc_val = []
                print(text)
                log.write(text)

        # Save final training statistics
        train_stats = {
            'loss_train': c_loss_train,
            'acc_train': c_acc_train,
            'loss_val': c_loss_val,
            'acc_val': c_acc_val,
            'epoch': epoch
        }

        torch.save(train_stats, str(save_path) + '.train')
        print('Training saved!')

        return net

    def _sweep_data(self, net, trials, idx, training=True):
        # Initialize batch processing variables
        n_batch = 0
        sample_size = idx.shape[0]
        idx_batch = np.arange(self.batch_size)
        
        # Extract and prepare input data
        in_arms = trials['in_arms']
        in_time = trials['in_time']
        labels = trials['labels']
        labels = torch.from_numpy(labels)
        labels = labels.to(torch.float32)
        inputs = torch.from_numpy(np.concatenate((in_arms, in_time), axis=1))

        # Initialize tracking variables
        loss_total = 0
        cor_total = 0
        denom_total = 0

        while True:
            # Prepare batch data
            inputs_batch = inputs[idx[idx_batch], :, :]
            labels_batch = labels[idx[idx_batch], :, :]

            # Set gradients based on training mode
            if training:
                net.requires_grad_(True)
            else:
                net.requires_grad_(False)

            # Forward pass
            results_batch = net(inputs_batch)

            # Calculate loss and accuracy
            loss_batch = self.loss_func(inputs_batch, results_batch, labels_batch)
            cor_total += self.acc_func(results_batch, labels_batch)

            denom_total += self.batch_size
            loss_total += loss_batch

            # Backward pass if training
            if training:
                net.zero_grad()
                optimizer = self.optimizer(net.parameters(), self.lr)
                loss_batch.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Update batch indices
            n_batch += 1
            idx_batch = (idx_batch + self.batch_size) % sample_size

            # Check if epoch complete
            if 0 in idx_batch:
                loss_total /= n_batch
                acc_total = cor_total / denom_total
                break

        return loss_total, acc_total
