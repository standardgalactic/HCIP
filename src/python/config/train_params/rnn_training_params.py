config_common = {
        'batch_size': 64,
        'sweep_max': 500,
        'epoch_max': 1e6,
        'lr': 5e-4,
        'optimizer': 'Adam',
        'val_size': 0.3,
        'stop_after': 5
    }


def config_self_supervised():
    config = {
        'loss_func': '_loss_self_supervised',
        'acc_func': '_acc_func'
    }
    config.update(config_common)
    return config


def config_ext_supervised():
    config = {
        'loss_func': '_loss_ext_supervised',
        'acc_func': '_acc_func'
    }
    config.update(config_common)
    return config
