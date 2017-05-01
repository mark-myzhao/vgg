"""Training and Testing configurations."""


class Config(object):
    """Config."""

    # util
    batch_size = 1
    initialize = True
    steps = "-1"
    gpu = '/gpu:0'

    # checkpoint path and filename
    logdir = "./log/train_log/"
    params_dir = "./params/"
    load_filename = "vgg" + '-' + steps
    save_filename = "vgg"

    # iterations config
    max_iteration = 100000
    checkpoint_iters = 2000
    summary_iters = 100
    validate_iters = 2000

    # image config
    points_num = 15
    channel_num = 3  # RGB
    fm_channel = points_num + 1
    origin_height = 212
    origin_width = 256
    img_height = 224  # img height for training
    img_width = 224  # img width for training
    is_color = True

    # random distortion
    degree = 15

    # solver config
    wd = 5e-4
    stddev = 5e-2
    use_fp16 = False
    moving_average_decay = 0.999

    def __init__(self):
        """Initializer."""
        pass
