

def get_config():
    config = {
        # Model architecture
        'seq_window_lengths': [8, 12],
        'smi_window_lengths': [4, 8],
        'num_windows': [32, 64, 96],
        'num_hidden': 1024,
        'num_classes': 1,

        # Input dimensions
        'max_seq_len': 1000,
        'max_smi_len': 100,

        # Training parameters
        'learning_rate': 0.001,
        'num_epoch': 100,
        'batch_size': 256,

        # Data handling
        'davis_path': '../data/davis/',
        'kiba_path': '../data/kiba/',
        'problem_type': 1,
        'binary_th': 0.0,

        # Output and logging
        'checkpoint_path': '',
        'log_dir': 'logs',

        # Model type
        'model_type': 'cnn',  # or 'attention'

        # Dataset specific
        'davis_convert_to_log': True,
        'kiba_convert_to_log': False,
    }

    return config


def print_config(config):
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n")
