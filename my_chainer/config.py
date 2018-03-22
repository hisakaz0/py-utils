

def set_train(config):
    config.cudnn_deterministic = True  # To make sure reproduction
    config.train               = True
    config.enable_backprop     = True
    config.type_check          = True
    config.autotune            = True
    config.use_cudnn           = 'always'

def set_val(config):
    config.cudnn_deterministic = True  # To make sure reproduction
    config.train               = False
    config.enable_backprop     = False
    config.type_check          = True
    config.autotune            = True
    config.use_cudnn           = 'always'
