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
        'davis_path': '/content/drive/MyDrive/BioInf-Final/data/davis/',
        'kiba_path': '/content/drive/MyDrive/BioInf-Final/data/kiba/',
        'problem_type': 1,
        'binary_th': 0.0,

        # Output and logging
        'checkpoint_path': '',
        'log_dir': 'logs',

        # Model type
        'model_type': 'cnn',  # 'cnn', 'lstm', 'lstm_attention', or 'transformer'

        # Dataset specific
        'davis_convert_to_log': True,
        'kiba_convert_to_log': False,

        # LSTM parameters
        'lstm_units': 128,
        'lstm_layers': 2,

        # Transformer parameters
        'num_heads': 8,
        'num_layers': 4,
        'ff_dim': 2048,
        'dropout_rate': 0.1,

        # Alphabet sizes
        'charsmiset_size': 64,  # Size of SMILES alphabet
        'charseqset_size': 26,  # Size of protein sequence alphabet
    }

    return config


def print_config(config):
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\n")
