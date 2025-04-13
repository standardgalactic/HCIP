# Common configuration parameters shared between supervised and self-supervised training
config_common = {
        'batch_size': 64,        # Number of samples per training batch
        'sweep_max': 500,        # Maximum number of sweeps through the data
        'epoch_max': 1e6,        # Maximum number of training epochs
        'lr': 5e-4,             # Learning rate
        'optimizer': 'Adam',     # Optimization algorithm
        'val_size': 0.3,        # Fraction of data used for validation
        'stop_after': 5         # Number of epochs without improvement before early stopping
    }


def config_self_supervised():
    """Returns configuration for self-supervised training.
    Uses internal model predictions for supervision."""
    config = {
        'loss_func': '_loss_self_supervised',  # Self-supervised loss function
        'acc_func': '_acc_func'                # Accuracy calculation function
    }
    config.update(config_common)
    return config


def config_ext_supervised():
    """Returns configuration for externally supervised training.
    Uses external ground truth labels for supervision."""
    config = {
        'loss_func': '_loss_ext_supervised',   # Externally supervised loss function
        'acc_func': '_acc_func'                # Accuracy calculation function
    }
    config.update(config_common)
    return config
