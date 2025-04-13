import copy
import scipy.stats as stats
import torch.nn as nn
import torch
import numpy as np


def _compute_likelihood(tm, arms):
    """
    Compute likelihood of timing match between temporal signal and arm lengths
    
    Args:
        tm: Temporal signal array (batch_size,) 
        arms: Arm lengths array (batch_size, n_arms)
        
    Returns:
        prs: Likelihood array (batch_size, n_arms) containing squared differences
             between temporal signal and normalized arm lengths
    """
    # Initialize likelihood array
    prs = torch.zeros((arms.shape[0], arms.shape[1]))
    
    # Compute likelihood for each arm
    for i in range(arms.shape[1]):
        arms_i = arms[:, i]
        # Calculate squared difference between temporal signal and normalized arm length
        # Normalize by 15% of arm length (0.15 * arms_i)
        prs[:, i] = torch.pow((torch.from_numpy(tm) - arms_i) / (0.15 * arms_i), 2)

    # Set likelihood to 0 when there is no temporal signal (tm == 0)
    prs[tm == 0, :] = 0

    return prs


class RNN_ATT_NOISE_TM1(nn.Module):
    """
    RNN with attention mechanism and noise injection after attention switch.
    This model processes temporal and spatial information from an H-maze task,
    with attention-based switching between maze arms and noise injection to simulate
    uncertainty in temporal estimation after switching attention.
    """

    def __init__(self, params_net):
        super().__init__()
        self.out_size = 2  # Number of arms per branch (left/right)

        self.nonlinearity = params_net['nonlinearity']  # Activation function type

        # Recurrent weights for maintaining hidden state
        self.W_rec = nn.Linear(in_features=params_net['hidden_size'], out_features=params_net['hidden_size'])

        # Output weights for generating attention keys
        self.W_key = nn.Linear(in_features=params_net['hidden_size'], out_features=self.out_size)

        # Input weights for processing arm lengths and timing signals
        # Input size is 6: 2 horizontal arms + 4 vertical arms
        self.W_linear_in = nn.Linear(in_features=6, out_features=params_net['hidden_size'])

        # Parameters for noise injection
        self.wb = params_net['task_data_generator'].wb  # Base width
        self.wb_inc = params_net['task_data_generator'].wb_inc  # Width increment

    def forward(self, inputs):
        inputs.requires_grad_(False)
        # Split input into arm lengths and timing signals
        in_arms = inputs[:, :-2, :]  # All arm lengths
        in_time = inputs[:, -2:, :]  # Two timing signals (tm1, tm2)

        in_time = in_time.detach().numpy()

        # Find timesteps when timing signals are presented
        idx_tm = np.argmax(in_time, axis=2)

        # Get dimensions
        n_timesteps = inputs.shape[2]
        batch_size = inputs.shape[0]
        n_hidden = self.W_rec.in_features

        # Initialize activation function
        f = eval('torch.nn.' + self.nonlinearity + '()')

        # Initialize output tensors
        keys = torch.zeros((batch_size, self.out_size, n_timesteps))  # Attention keys
        prs = torch.zeros((batch_size, self.out_size * 2, n_timesteps))  # Likelihoods
        hs = torch.zeros((batch_size, n_hidden, n_timesteps))  # Hidden states

        # Initialize tracking variables
        key_h_pre = torch.ones(batch_size)  # Previous horizontal attention key
        key_h_pre.requires_grad_(True)
        h_pre = torch.zeros((batch_size, n_hidden))  # Previous hidden state
        side_h_pre = key_h_pre > 0.5  # Previous side selection (True=left, False=right)

        wb_inc = self.wb_inc  # Width increment for noise injection

        # Process each timestep
        for t in range(1, n_timesteps):
            # Get current arm lengths
            arms_pri_t = in_arms[:, :2, t]  # Horizontal arms
            arms_sec_t = in_arms[:, 2:, t]  # Vertical arms

            # Determine current side selection
            side_h_t = key_h_pre > 0.5

            # Weight vertical arms based on attention
            arms_l_t = arms_sec_t[:, :2] * torch.unsqueeze(key_h_pre, axis=1)  # Left vertical arms
            arms_r_t = arms_sec_t[:, 2:] * torch.unsqueeze(1 - key_h_pre, axis=1)  # Right vertical arms
            arms_sec_sum_t = arms_l_t + arms_r_t  # Combined weighted vertical arms

            # Check timing conditions for noise injection
            tm1_done = t > idx_tm[:, 0]  # After first timing signal
            tm1 = np.expand_dims(in_time[:, 0, t], axis=1)
            closer_ts1 = np.argmin(abs(arms_pri_t.detach().numpy() - tm1), axis=1) == 0
            away_from = ~np.equal(side_h_t.detach().numpy(), closer_ts1)

            # Inject noise when attention switches sides after tm1
            switched_t = torch.squeeze(side_h_t != side_h_pre) & tm1_done & away_from
            switched_t = np.where(switched_t)[0]
            if len(switched_t) != 0:
                # Add Gaussian noise proportional to timing value
                in_time[switched_t, 1, t:] = in_time[switched_t, 1, t:] \
                                             + np.random.normal(0, abs(wb_inc * in_time[switched_t, 1, t]))[:,
                                               np.newaxis]

            # Compute likelihoods between timing signals and arm lengths
            pr1_t = _compute_likelihood(in_time[:, 0, t], arms_pri_t)
            pr2_t = _compute_likelihood(in_time[:, 1, t], arms_sec_sum_t)
            prs_t = torch.cat((pr1_t, pr2_t), dim=1).float()

            # Prepare input for RNN
            arms_t = torch.cat((arms_pri_t, arms_sec_sum_t), dim=1)
            inputs_t = torch.cat((arms_t, torch.from_numpy(in_time[:, :, t])), dim=1).float()
            
            # Update hidden state
            h_t = f(self.W_linear_in(inputs_t) + self.W_rec(h_pre))

            # Generate attention keys
            key_t = self.W_key(h_t)
            key_t = 1 / (1 + torch.exp(-1 * key_t))  # Sigmoid activation

            # Store current timestep outputs
            hs[:, :, t] = h_t
            prs[:, :, t] = prs_t
            keys[:, :, t] = key_t

            # Update tracking variables
            key_h_pre = key_t[:, 0]
            h_pre = h_t
            side_h_pre = side_h_t

        # Return all computed values
        results = {
            'keys': keys,          # Attention keys for each timestep
            'prs': prs,            # Likelihoods for each timestep
            'hs': hs,              # Hidden states for each timestep
            'tm_modified': in_time # Timing signals with injected noise
        }
        return results


class RNN_ATT_noise(nn.Module):
    """
    RNN with attention mechanism but no noise injection.
    This class implements a recurrent neural network that uses an attention mechanism
    to focus on different parts of the input, but without noise injection.
    """

    def __init__(self, params_net):
        super().__init__()
        self.out_size = 2  # Number of arms per branch

        self.nonlinearity = params_net['nonlinearity']  # Activation function type

        # Recurrent weights for hidden state transitions
        self.W_rec = nn.Linear(in_features=params_net['hidden_size'], out_features=params_net['hidden_size'])

        # Output weights for generating attention keys
        self.W_key = nn.Linear(in_features=params_net['hidden_size'], out_features=self.out_size)

        # Input weights for processing raw inputs
        self.W_linear_in = nn.Linear(in_features=6, out_features=params_net['hidden_size'])

        # Base width and increment parameters from task generator
        self.wb = params_net['task_data_generator'].wb
        self.wb_inc = params_net['task_data_generator'].wb_inc

    def forward(self, inputs):
        inputs.requires_grad_(False)  # Disable gradient computation for inputs
        in_arms = inputs[:, :-2, :]   # Extract arm length inputs
        in_time = inputs[:, -2:, :]   # Extract timing signal inputs

        in_time = in_time.detach().numpy()  # Convert timing signals to numpy array

        # Find timesteps when timing signals are presented
        idx_tm = np.argmax(in_time, axis=2)

        # Get dimensions
        n_timesteps = inputs.shape[2]
        batch_size = inputs.shape[0]
        n_hidden = self.W_rec.in_features

        # Set up activation function
        f = eval('torch.nn.' + self.nonlinearity + '()')

        # Initialize output tensors
        keys = torch.zeros((batch_size, self.out_size, n_timesteps))  # Attention keys
        prs = torch.zeros((batch_size, self.out_size * 2, n_timesteps))  # Likelihoods
        hs = torch.zeros((batch_size, n_hidden, n_timesteps))  # Hidden states

        # Initialize tracking variables
        key_h_pre = torch.ones(batch_size)  # Previous horizontal attention key
        key_h_pre.requires_grad_(True)
        h_pre = torch.zeros((batch_size, n_hidden))  # Previous hidden state
        side_h_pre = key_h_pre > 0.5  # Previous attended side

        wb_inc = self.wb_inc  # Width increment parameter

        # Process each timestep
        for t in range(1, n_timesteps):
            # Extract current arm inputs
            arms_pri_t = in_arms[:, :2, t]  # Horizontal arms
            arms_sec_t = in_arms[:, 2:, t]  # Vertical arms

            # Determine current attended side
            side_h_t = key_h_pre > 0.5

            # Apply attention to vertical arms
            arms_l_t = arms_sec_t[:, :2] * torch.unsqueeze(key_h_pre, axis=1)  # Left arms
            arms_r_t = arms_sec_t[:, 2:] * torch.unsqueeze(1 - key_h_pre, axis=1)  # Right arms
            arms_sec_sum_t = arms_l_t + arms_r_t  # Combine attended arms

            # Compute likelihoods between timing signals and arm lengths
            pr1_t = _compute_likelihood(in_time[:, 0, t], arms_pri_t)
            pr2_t = _compute_likelihood(in_time[:, 1, t], arms_sec_sum_t)
            prs_t = torch.cat((pr1_t, pr2_t), dim=1).float()

            # Prepare input for RNN
            arms_t = torch.cat((arms_pri_t, arms_sec_sum_t), dim=1)
            inputs_t = torch.cat((arms_t, torch.from_numpy(in_time[:, :, t])), dim=1).float()
            
            # Update hidden state
            h_t = f(self.W_linear_in(inputs_t) + self.W_rec(h_pre))

            # Generate attention keys and apply sigmoid activation
            key_t = self.W_key(h_t)
            key_t = 1 / (1 + torch.exp(-1 * key_t))

            # Store current timestep outputs
            hs[:, :, t] = h_t
            prs[:, :, t] = prs_t
            keys[:, :, t] = key_t

            # Update tracking variables for next timestep
            key_h_pre = key_t[:, 0]
            h_pre = h_t
            side_h_pre = side_h_t

        # Return dictionary of results
        results = {
            'keys': keys,          # Attention keys for each timestep
            'prs': prs,            # Likelihoods for each timestep
            'hs': hs,              # Hidden states for each timestep
            'tm_modified': in_time # Original timing signals
        }
        return results


class RNN_att_noise(nn.Module):
    """
    RNN without attention mechanism or noise injection.
    Basic RNN model that processes arm lengths and timing signals from an H-maze task.
    """

    def __init__(self, params_net):
        super().__init__()
        self.out_size = 2  # Number of arms per branch (left/right)

        self.nonlinearity = params_net['nonlinearity']  # Activation function type

        # Recurrent weights for maintaining hidden state
        self.W_rec = nn.Linear(in_features=params_net['hidden_size'], out_features=params_net['hidden_size'])

        # Output weights for generating keys
        self.W_key = nn.Linear(in_features=params_net['hidden_size'], out_features=self.out_size)

        # Input weights for processing arm lengths and timing signals
        # Input size is 8: 2 horizontal arms + 4 vertical arms + 2 timing signals
        self.W_linear_in = nn.Linear(in_features=8, out_features=params_net['hidden_size'])

        # Parameters for noise injection (unused in this model)
        self.wb = params_net['task_data_generator'].wb  # Base width
        self.wb_inc = params_net['task_data_generator'].wb_inc  # Width increment

    def forward(self, inputs):
        inputs.requires_grad_(False)
        # Split input into arm lengths and timing signals
        in_arms = inputs[:, :-2, :]  # All arm lengths
        in_time = inputs[:, -2:, :]  # Two timing signals (tm1, tm2)

        in_time = in_time.detach().numpy()

        # Find timesteps when timing signals are presented
        idx_tm = np.argmax(in_time, axis=2)

        # Get dimensions
        n_timesteps = inputs.shape[2]
        batch_size = inputs.shape[0]
        n_hidden = self.W_rec.in_features

        # Initialize activation function
        f = eval('torch.nn.' + self.nonlinearity + '()')

        # Initialize output tensors
        keys = torch.zeros((batch_size, self.out_size, n_timesteps))  # Keys for each timestep
        prs = torch.zeros((batch_size, self.out_size * 2, n_timesteps))  # Likelihoods for each timestep
        hs = torch.zeros((batch_size, n_hidden, n_timesteps))  # Hidden states

        # Initialize tracking variables
        key_h_pre = torch.ones(batch_size)  # Previous horizontal key
        key_h_pre.requires_grad_(True)
        h_pre = torch.zeros((batch_size, n_hidden))  # Previous hidden state

        # Process each timestep
        for t in range(1, n_timesteps):
            # Extract current arm inputs
            arms_pri_t = in_arms[:, :2, t]  # Horizontal arms
            arms_sec_t = in_arms[:, 2:, t]  # Vertical arms

            # Determine current attended side
            side_h_t = key_h_pre > 0.5

            # Apply attention to vertical arms
            arms_l_t = arms_sec_t[:, :2] * torch.unsqueeze(key_h_pre, axis=1)  # Left arms
            arms_r_t = arms_sec_t[:, 2:] * torch.unsqueeze(1 - key_h_pre, axis=1)  # Right arms
            arms_sec_sum_t = arms_l_t + arms_r_t  # Combine attended arms

            # Compute likelihoods between timing signals and arm lengths
            pr1_t = _compute_likelihood(in_time[:, 0, t], arms_pri_t)
            pr2_t = _compute_likelihood(in_time[:, 1, t], arms_sec_sum_t)
            prs_t = torch.cat((pr1_t, pr2_t), dim=1).float()

            # Prepare input for RNN
            inputs_t = inputs[:, :, t].float()
            
            # Update hidden state
            h_t = f(self.W_linear_in(inputs_t) + self.W_rec(h_pre))

            # Generate keys and apply sigmoid activation
            key_t = self.W_key(h_t)
            key_t = 1 / (1 + torch.exp(-1 * key_t))

            # Store current timestep outputs
            hs[:, :, t] = h_t
            prs[:, :, t] = prs_t
            keys[:, :, t] = key_t

            # Update tracking variables for next timestep
            key_h_pre = key_t[:, 0]
            h_pre = h_t
            side_h_pre = side_h_t

        # Return dictionary of results
        results = {
            'keys': keys,          # Keys for each timestep
            'prs': prs,            # Likelihoods for each timestep
            'hs': hs,              # Hidden states for each timestep
            'tm_modified': in_time # Original timing signals
        }
        return results


class RNN_att_NOISE(nn.Module):
    """
    RNN without attention mechanism but with noise injection after attention switch.
    This model adds noise to temporal signals when attention switches between arms.
    """

    def __init__(self, params_net):
        super().__init__()
        self.out_size = 2  # Number of arms per branch (left/right)

        self.nonlinearity = params_net['nonlinearity']  # Activation function type

        # Recurrent weights for maintaining hidden state
        self.W_rec = nn.Linear(in_features=params_net['hidden_size'], out_features=params_net['hidden_size'])

        # Output weights for generating keys
        self.W_key = nn.Linear(in_features=params_net['hidden_size'], out_features=self.out_size)

        # Input weights for processing arm lengths and timing signals
        # Input size is 8: 2 horizontal arms + 4 vertical arms + 2 timing signals
        self.W_linear_in = nn.Linear(in_features=8, out_features=params_net['hidden_size'])

        # Parameters for noise injection
        self.wb = params_net['task_data_generator'].wb  # Base width
        self.wb_inc = params_net['task_data_generator'].wb_inc  # Width increment

    def forward(self, inputs):
        inputs.requires_grad_(False)
        # Split input into arm lengths and timing signals
        in_arms = inputs[:, :-2, :]  # All arm lengths
        in_time = inputs[:, -2:, :]  # Two timing signals (tm1, tm2)

        in_time = in_time.detach().numpy()

        # Find timesteps when timing signals are presented
        idx_tm = np.argmax(in_time, axis=2)

        # Get dimensions
        n_timesteps = inputs.shape[2]
        batch_size = inputs.shape[0]
        n_hidden = self.W_rec.in_features

        # Initialize activation function
        f = eval('torch.nn.' + self.nonlinearity + '()')

        # Initialize output tensors
        keys = torch.zeros((batch_size, self.out_size, n_timesteps))  # Keys for each timestep
        prs = torch.zeros((batch_size, self.out_size * 2, n_timesteps))  # Likelihoods for each timestep
        hs = torch.zeros((batch_size, n_hidden, n_timesteps))  # Hidden states

        # Initialize tracking variables
        key_h_pre = torch.ones(batch_size)  # Previous horizontal key
        key_h_pre.requires_grad_(True)
        h_pre = torch.zeros((batch_size, n_hidden))  # Previous hidden state
        side_h_pre = key_h_pre > 0.5  # Previous attended side

        wb_inc = self.wb_inc  # Noise width increment

        # Process each timestep
        for t in range(1, n_timesteps):
            # Extract current arm inputs
            arms_pri_t = in_arms[:, :2, t]  # Horizontal arms
            arms_sec_t = in_arms[:, 2:, t]  # Vertical arms

            # Determine current attended side
            side_h_t = key_h_pre > 0.5

            # Apply attention to vertical arms
            arms_l_t = arms_sec_t[:, :2] * torch.unsqueeze(key_h_pre, axis=1)  # Left arms
            arms_r_t = arms_sec_t[:, 2:] * torch.unsqueeze(1 - key_h_pre, axis=1)  # Right arms
            arms_sec_sum_t = arms_l_t + arms_r_t  # Combine attended arms

            # Check conditions for noise injection
            tm1_done = t > idx_tm[:, 0]  # After first timing signal
            tm1 = np.expand_dims(in_time[:, 0, t], axis=1)
            closer_ts1 = np.argmin(abs(arms_pri_t.detach().numpy() - tm1), axis=1) == 0
            away_from = ~np.equal(side_h_t.detach().numpy(), closer_ts1)
            
            # Detect attention switches after tm2 presentation
            switched_t = torch.squeeze(side_h_t != side_h_pre) & tm1_done & away_from
            switched_t = np.where(switched_t)[0]

            # Inject noise when attention switches
            if len(switched_t) != 0:
                noise = np.random.normal(0, abs(wb_inc * in_time[switched_t, 1, t]))[:, np.newaxis]
                in_time[switched_t, 1, t:] = in_time[switched_t, 1, t:] + noise

            # Compute likelihoods between timing signals and arm lengths
            pr1_t = _compute_likelihood(in_time[:, 0, t], arms_pri_t)
            pr2_t = _compute_likelihood(in_time[:, 1, t], arms_sec_sum_t)
            prs_t = torch.cat((pr1_t, pr2_t), dim=1).float()

            # Prepare input for RNN
            arms_t = torch.cat((arms_pri_t, arms_sec_sum_t), dim=1)
            inputs_t = torch.cat((in_arms[:, :, t], torch.from_numpy(in_time[:, :, t])), dim=1).float()

            # Update hidden state
            h_t = f(self.W_linear_in(inputs_t) + self.W_rec(h_pre))

            # Generate keys and apply sigmoid activation
            key_t = self.W_key(h_t)
            key_t = 1 / (1 + torch.exp(-1 * key_t))

            # Store current timestep outputs
            hs[:, :, t] = h_t
            prs[:, :, t] = prs_t
            keys[:, :, t] = key_t

            # Update tracking variables for next timestep
            key_h_pre = key_t[:, 0]
            h_pre = h_t
            side_h_pre = side_h_t

        # Return dictionary of results
        results = {
            'keys': keys,          # Keys for each timestep
            'prs': prs,            # Likelihoods for each timestep
            'hs': hs,              # Hidden states for each timestep
            'tm_modified': in_time # Modified timing signals with injected noise
        }
        return results