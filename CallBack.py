import math
from keras.callbacks import LearningRateScheduler, EarlyStopping


def GetCallback(config):
    callbacks = []
    if config["epoch_lr_drop"]:
        ## reduce learning rate epoch
        def step_decay(epoch, init_lr, drop, epochs_drop):
            initial_lrate = init_lr
            lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
            return lrate

        from functools import partial

        step_decay_part = partial(step_decay, init_lr=config["learning_rate"], drop=config["reduce_lr_drop"],
                                  epochs_drop=config["epoch"])
        lr_callback = LearningRateScheduler(step_decay_part, verbose=0)
        callbacks.append(lr_callback)

    if config["early_stopping"]:
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='min')
        callbacks.append(early_stopping)

    return callbacks