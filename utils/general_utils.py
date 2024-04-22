def update_config_samples(config, mode, length):
    if 'N_SAMPLES' not in dir(config):
        config.N_SAMPLES = dict()
        config.N_STEPS_PER_EPOCH = dict()
        config.N_STEPS = dict()
        
    config.N_SAMPLES[mode] = length
    config.N_STEPS_PER_EPOCH[mode] = (config.N_SAMPLES[mode] // config.BATCH_SIZE)
    config.N_STEPS[mode] = config.N_STEPS_PER_EPOCH[mode] * config.N_EPOCHS + 1
    return config