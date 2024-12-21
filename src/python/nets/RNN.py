import copy
import scipy.stats as stats
import torch.nn as nn
import torch
import numpy as np


def _compute_likelihood(tm, arms):
    prs = torch.zeros((arms.shape[0], arms.shape[1]))
    for i in range(arms.shape[1]):
        arms_i = arms[:, i]
        prs[:, i] = torch.pow((torch.from_numpy(tm) - arms_i) / (0.15 * arms_i), 2)

    # don't need to worry about pr when there is no tm
    prs[tm == 0, :] = 0

    return prs


class RNN_ATT_NOISE_TM1(nn.Module):
    """

    """

    def __init__(self, params_net):
        super().__init__()
        self.out_size = 2  # arms per branch

        self.nonlinearity = params_net['nonlinearity']

        # recurrent weights
        self.W_rec = nn.Linear(in_features=params_net['hidden_size'], out_features=params_net['hidden_size'])

        # output weights for two choices
        self.W_key = nn.Linear(in_features=params_net['hidden_size'], out_features=self.out_size)

        # input weights
        self.W_linear_in = nn.Linear(in_features=6, out_features=params_net['hidden_size'])

        self.wb = params_net['task_data_generator'].wb
        self.wb_inc = params_net['task_data_generator'].wb_inc

    def forward(self, inputs):
        inputs.requires_grad_(False)
        in_arms = inputs[:, :-2, :]
        in_time = inputs[:, -2:, :]

        in_time = in_time.detach().numpy()

        # when tm1 and tm2 are presented
        idx_tm = np.argmax(in_time, axis=2)

        n_timesteps = inputs.shape[2]
        batch_size = inputs.shape[0]
        n_hidden = self.W_rec.in_features

        f = eval('torch.nn.' + self.nonlinearity + '()')

        keys = torch.zeros((batch_size, self.out_size, n_timesteps))
        prs = torch.zeros((batch_size, self.out_size * 2, n_timesteps))
        # hidden states
        hs = torch.zeros((batch_size, n_hidden, n_timesteps))

        # keep track of horizontal key
        key_h_pre = torch.ones(batch_size)
        key_h_pre.requires_grad_(True)
        h_pre = torch.zeros((batch_size, n_hidden))
        side_h_pre = key_h_pre > 0.5

        wb_inc = self.wb_inc

        # # keep track of if key_h switched size
        # switched = np.zeros(batch_size)

        for t in range(1, n_timesteps):
            # two horizontal arms
            arms_pri_t = in_arms[:, :2, t]
            # four vertical arms
            arms_sec_t = in_arms[:, 2:, t]

            side_h_t = key_h_pre > 0.5

            arms_l_t = arms_sec_t[:, :2] * torch.unsqueeze(key_h_pre, axis=1)
            arms_r_t = arms_sec_t[:, 2:] * torch.unsqueeze(1 - key_h_pre, axis=1)
            # linear sum of two sides
            arms_sec_sum_t = arms_l_t + arms_r_t

            # propagate tm(t-1) to tm(t)
            # only do after tm1
            tm1_done = t > idx_tm[:, 0]
            # if sum(tm1_done) != 0:
            #     in_time[tm1_done, :, t] = in_time[tm1_done, :, t - 1]
            tm1 = np.expand_dims(in_time[:, 0, t], axis=1)
            closer_ts1 = np.argmin(abs(arms_pri_t.detach().numpy() - tm1), axis=1) == 0
            away_from = ~np.equal(side_h_t.detach().numpy(), closer_ts1)
            # if switched side after tm2 being presented, inject noise to tm
            switched_t = torch.squeeze(side_h_t != side_h_pre) & tm1_done & away_from
            # switched = np.logical_or(switched, switched_t.detach().numpy())
            switched_t = np.where(switched_t)[0]
            if len(switched_t) != 0:
                # np.random.seed()
                in_time[switched_t, 1, t:] = in_time[switched_t, 1, t:] \
                                             + np.random.normal(0, abs(wb_inc * in_time[switched_t, 1, t]))[:,
                                               np.newaxis]

            pr1_t = _compute_likelihood(in_time[:, 0, t], arms_pri_t)
            pr2_t = _compute_likelihood(in_time[:, 1, t], arms_sec_sum_t)
            prs_t = torch.cat((pr1_t, pr2_t), dim=1).float()

            arms_t = torch.cat((arms_pri_t, arms_sec_sum_t), dim=1)
            inputs_t = torch.cat((arms_t, torch.from_numpy(in_time[:, :, t])), dim=1).float()
            h_t = f(self.W_linear_in(inputs_t) + self.W_rec(h_pre))

            key_t = self.W_key(h_t)
            key_t = 1 / (1 + torch.exp(-1 * key_t))

            hs[:, :, t] = h_t
            prs[:, :, t] = prs_t
            keys[:, :, t] = key_t

            key_h_pre = key_t[:, 0]
            h_pre = h_t
            side_h_pre = side_h_t

        results = {
            'keys': keys,
            'prs': prs,
            'hs': hs,
            'tm_modified': in_time
        }
        return results


class RNN_ATT_noise(nn.Module):
    """

    """

    def __init__(self, params_net):
        super().__init__()
        self.out_size = 2  # arms per branch

        self.nonlinearity = params_net['nonlinearity']

        # recurrent weights
        self.W_rec = nn.Linear(in_features=params_net['hidden_size'], out_features=params_net['hidden_size'])

        # output weights for two choices
        self.W_key = nn.Linear(in_features=params_net['hidden_size'], out_features=self.out_size)

        # input weights
        self.W_linear_in = nn.Linear(in_features=6, out_features=params_net['hidden_size'])

        self.wb = params_net['task_data_generator'].wb
        self.wb_inc = params_net['task_data_generator'].wb_inc

    def forward(self, inputs):
        inputs.requires_grad_(False)
        in_arms = inputs[:, :-2, :]
        in_time = inputs[:, -2:, :]

        in_time = in_time.detach().numpy()

        # when tm1 and tm2 are presented
        idx_tm = np.argmax(in_time, axis=2)

        n_timesteps = inputs.shape[2]
        batch_size = inputs.shape[0]
        n_hidden = self.W_rec.in_features

        f = eval('torch.nn.' + self.nonlinearity + '()')

        keys = torch.zeros((batch_size, self.out_size, n_timesteps))
        prs = torch.zeros((batch_size, self.out_size * 2, n_timesteps))
        # hidden states
        hs = torch.zeros((batch_size, n_hidden, n_timesteps))

        # keep track of horizontal key
        key_h_pre = torch.ones(batch_size)
        key_h_pre.requires_grad_(True)
        h_pre = torch.zeros((batch_size, n_hidden))
        side_h_pre = key_h_pre > 0.5

        wb_inc = self.wb_inc

        # # keep track of if key_h switched size
        # switched = np.zeros(batch_size)

        for t in range(1, n_timesteps):
            # two horizontal arms
            arms_pri_t = in_arms[:, :2, t]
            # four vertical arms
            arms_sec_t = in_arms[:, 2:, t]

            side_h_t = key_h_pre > 0.5

            arms_l_t = arms_sec_t[:, :2] * torch.unsqueeze(key_h_pre, axis=1)
            arms_r_t = arms_sec_t[:, 2:] * torch.unsqueeze(1 - key_h_pre, axis=1)
            # linear sum of two sides
            arms_sec_sum_t = arms_l_t + arms_r_t

            # # propagate tm(t-1) to tm(t)
            # # only do after tm1
            # tm1_done = t > idx_tm[:, 0]
            # # if sum(tm1_done) != 0:
            # #     in_time[tm1_done, :, t] = in_time[tm1_done, :, t - 1]
            # tm1 = np.expand_dims(in_time[:, 0, t], axis=1)
            # closer_ts1 = np.argmin(abs(arms_pri_t.detach().numpy() - tm1), axis=1) == 0
            # away_from = ~np.equal(side_h_t.detach().numpy(), closer_ts1)
            # # if switched side after tm2 being presented, inject noise to tm
            # switched_t = torch.squeeze(side_h_t != side_h_pre) & tm1_done & away_from
            # # switched = np.logical_or(switched, switched_t.detach().numpy())
            # switched_t = np.where(switched_t)[0]
            # if len(switched_t) != 0:
            #     # np.random.seed()
            #     in_time[switched_t, 1, t:] = in_time[switched_t, 1, t:] \
            #         + np.random.normal(0, abs(wb_inc * in_time[switched_t, 1, t]))[:, np.newaxis]

            pr1_t = _compute_likelihood(in_time[:, 0, t], arms_pri_t)
            pr2_t = _compute_likelihood(in_time[:, 1, t], arms_sec_sum_t)
            prs_t = torch.cat((pr1_t, pr2_t), dim=1).float()

            arms_t = torch.cat((arms_pri_t, arms_sec_sum_t), dim=1)
            inputs_t = torch.cat((arms_t, torch.from_numpy(in_time[:, :, t])), dim=1).float()
            h_t = f(self.W_linear_in(inputs_t) + self.W_rec(h_pre))

            key_t = self.W_key(h_t)
            key_t = 1 / (1 + torch.exp(-1 * key_t))

            hs[:, :, t] = h_t
            prs[:, :, t] = prs_t
            keys[:, :, t] = key_t

            key_h_pre = key_t[:, 0]
            h_pre = h_t
            side_h_pre = side_h_t

        results = {
            'keys': keys,
            'prs': prs,
            'hs': hs,
            'tm_modified': in_time
        }
        return results


class RNN_att_noise(nn.Module):
    """

    """

    def __init__(self, params_net):
        super().__init__()
        self.out_size = 2  # arms per branch

        self.nonlinearity = params_net['nonlinearity']

        # recurrent weights
        self.W_rec = nn.Linear(in_features=params_net['hidden_size'], out_features=params_net['hidden_size'])

        # output weights for two choices
        self.W_key = nn.Linear(in_features=params_net['hidden_size'], out_features=self.out_size)

        # input weights
        self.W_linear_in = nn.Linear(in_features=8, out_features=params_net['hidden_size'])

        self.wb = params_net['task_data_generator'].wb
        self.wb_inc = params_net['task_data_generator'].wb_inc

    def forward(self, inputs):
        inputs.requires_grad_(False)
        in_arms = inputs[:, :-2, :]
        in_time = inputs[:, -2:, :]

        in_time = in_time.detach().numpy()

        # when tm1 and tm2 are presented
        idx_tm = np.argmax(in_time, axis=2)

        n_timesteps = inputs.shape[2]
        batch_size = inputs.shape[0]
        n_hidden = self.W_rec.in_features

        f = eval('torch.nn.' + self.nonlinearity + '()')

        keys = torch.zeros((batch_size, self.out_size, n_timesteps))
        prs = torch.zeros((batch_size, self.out_size * 2, n_timesteps))
        # hidden states
        hs = torch.zeros((batch_size, n_hidden, n_timesteps))

        # keep track of horizontal key
        key_h_pre = torch.ones(batch_size)
        key_h_pre.requires_grad_(True)
        h_pre = torch.zeros((batch_size, n_hidden))


        # # keep track of if key_h switched size
        # switched = np.zeros(batch_size)

        for t in range(1, n_timesteps):
            # two horizontal arms
            arms_pri_t = in_arms[:, :2, t]
            # four vertical arms
            arms_sec_t = in_arms[:, 2:, t]

            side_h_t = key_h_pre > 0.5

            arms_l_t = arms_sec_t[:, :2] * torch.unsqueeze(key_h_pre, axis=1)
            arms_r_t = arms_sec_t[:, 2:] * torch.unsqueeze(1 - key_h_pre, axis=1)
            # linear sum of two sides
            arms_sec_sum_t = arms_l_t + arms_r_t

            pr1_t = _compute_likelihood(in_time[:, 0, t], arms_pri_t)
            pr2_t = _compute_likelihood(in_time[:, 1, t], arms_sec_sum_t)
            prs_t = torch.cat((pr1_t, pr2_t), dim=1).float()

            # arms_t = torch.cat((arms_pri_t, arms_sec_sum_t), dim=1)

            # inputs_t = torch.cat((arms_t, torch.from_numpy(in_time[:, :, t])), dim=1).float()
            inputs_t = inputs[:, :, t].float()
            h_t = f(self.W_linear_in(inputs_t) + self.W_rec(h_pre))

            key_t = self.W_key(h_t)
            key_t = 1 / (1 + torch.exp(-1 * key_t))

            hs[:, :, t] = h_t
            prs[:, :, t] = prs_t
            keys[:, :, t] = key_t

            key_h_pre = key_t[:, 0]
            h_pre = h_t
            side_h_pre = side_h_t

        results = {
            'keys': keys,
            'prs': prs,
            'hs': hs,
            'tm_modified': in_time
        }
        return results


class RNN_att_NOISE(nn.Module):
    """

    """

    def __init__(self, params_net):
        super().__init__()
        self.out_size = 2  # arms per branch

        self.nonlinearity = params_net['nonlinearity']

        # recurrent weights
        self.W_rec = nn.Linear(in_features=params_net['hidden_size'], out_features=params_net['hidden_size'])

        # output weights for two choices
        self.W_key = nn.Linear(in_features=params_net['hidden_size'], out_features=self.out_size)

        # input weights
        self.W_linear_in = nn.Linear(in_features=8, out_features=params_net['hidden_size'])

        self.wb = params_net['task_data_generator'].wb
        self.wb_inc = params_net['task_data_generator'].wb_inc

    def forward(self, inputs):
        inputs.requires_grad_(False)
        in_arms = inputs[:, :-2, :]
        in_time = inputs[:, -2:, :]

        in_time = in_time.detach().numpy()

        # when tm1 and tm2 are presented
        idx_tm = np.argmax(in_time, axis=2)

        n_timesteps = inputs.shape[2]
        batch_size = inputs.shape[0]
        n_hidden = self.W_rec.in_features

        f = eval('torch.nn.' + self.nonlinearity + '()')

        keys = torch.zeros((batch_size, self.out_size, n_timesteps))
        prs = torch.zeros((batch_size, self.out_size * 2, n_timesteps))
        # hidden states
        hs = torch.zeros((batch_size, n_hidden, n_timesteps))

        # keep track of horizontal key
        key_h_pre = torch.ones(batch_size)
        key_h_pre.requires_grad_(True)
        h_pre = torch.zeros((batch_size, n_hidden))
        side_h_pre = key_h_pre > 0.5

        wb_inc = self.wb_inc

        # # keep track of if key_h switched size
        # switched = np.zeros(batch_size)

        for t in range(1, n_timesteps):
            # two horizontal arms
            arms_pri_t = in_arms[:, :2, t]
            # four vertical arms
            arms_sec_t = in_arms[:, 2:, t]

            side_h_t = key_h_pre > 0.5

            arms_l_t = arms_sec_t[:, :2] * torch.unsqueeze(key_h_pre, axis=1)
            arms_r_t = arms_sec_t[:, 2:] * torch.unsqueeze(1 - key_h_pre, axis=1)
            # linear sum of two sides
            arms_sec_sum_t = arms_l_t + arms_r_t

            # propagate tm(t-1) to tm(t)
            # only do after tm1
            tm1_done = t > idx_tm[:, 0]
            # if sum(tm1_done) != 0:
            #     in_time[tm1_done, :, t] = in_time[tm1_done, :, t - 1]
            tm1 = np.expand_dims(in_time[:, 0, t], axis=1)
            closer_ts1 = np.argmin(abs(arms_pri_t.detach().numpy() - tm1), axis=1) == 0
            away_from = ~np.equal(side_h_t.detach().numpy(), closer_ts1)
            # if switched side after tm2 being presented, inject noise to tm
            switched_t = torch.squeeze(side_h_t != side_h_pre) & tm1_done & away_from
            # switched = np.logical_or(switched, switched_t.detach().numpy())
            switched_t = np.where(switched_t)[0]
            if len(switched_t) != 0:
                # np.random.seed()
                in_time[switched_t, 1, t:] = in_time[switched_t, 1, t:] \
                                             + np.random.normal(0, abs(wb_inc * in_time[switched_t, 1, t]))[:,
                                               np.newaxis]

            pr1_t = _compute_likelihood(in_time[:, 0, t], arms_pri_t)
            pr2_t = _compute_likelihood(in_time[:, 1, t], arms_sec_sum_t)
            prs_t = torch.cat((pr1_t, pr2_t), dim=1).float()

            arms_t = torch.cat((arms_pri_t, arms_sec_sum_t), dim=1)
            inputs_t = torch.cat((in_arms[:, :, t], torch.from_numpy(in_time[:, :, t])), dim=1).float()
            # inputs_t = inputs[:, :, t].float()

            h_t = f(self.W_linear_in(inputs_t) + self.W_rec(h_pre))

            key_t = self.W_key(h_t)
            key_t = 1 / (1 + torch.exp(-1 * key_t))

            hs[:, :, t] = h_t
            prs[:, :, t] = prs_t
            keys[:, :, t] = key_t

            key_h_pre = key_t[:, 0]
            h_pre = h_t
            side_h_pre = side_h_t

        results = {
            'keys': keys,
            'prs': prs,
            'hs': hs,
            'tm_modified': in_time
        }
        return results