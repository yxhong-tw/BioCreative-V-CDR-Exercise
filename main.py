from do_test import do_test
from do_train import do_train
from initialize import initialize
from utils import initialize_logger


configs = {}

# Checkpoint settings
configs['checkpoint_epoch'] = -1
configs['load_checkpoint'] = False
configs['save_time'] = 10

# Data settings
configs['dataloader_shuffle'] = True
configs['dataset_name'] = 'BC5CDR_Dataset'
configs['formatter_name'] = 'BC5CDR_Formatter'
configs['label_file_path'] = 'data/label.json'

# General settings
configs['logging_time'] = 25
configs['mode'] = 'train'
configs['model_name'] = 'LM_ForTokenClassification'
configs['num_labels'] = 5
configs['version'] = 'baseline-0'
configs['version_message'] = ''

# Model settings
# configs['freeze_lm'] = False
configs['lm_path'] = 'bert-base-uncased'
# configs['lm_path'] = 'microsoft/deberta-v3-base'

# Optimizer settings
configs['optimizer_lr'] = 1e-5
configs['optimizer_name'] = 'Adam'

# Scheduler settings
configs['scheduler_factor'] = 0.1
configs['scheduler_mode'] = 'min'
configs['scheduler_name'] = 'ReduceLROnPlateau'
configs['scheduler_patience'] = 2
configs['scheduler_verbose'] = True

# Training settings
configs['batch_size'] = 64
configs['epoch'] = 10

# Unused settings
# configs['batch_first'] = True
# configs['gru_bidirectional'] = True
# configs['gru_hidden_size'] = 768
# configs['gru_num_layers'] = 1
# configs['lm_hidden_size'] = 768


def main():
    logger = initialize_logger(configs=configs)
    parameters = initialize(configs=configs, mode=configs['mode'])

    if configs['mode'] == 'test':
        do_test(configs=configs, parameters=parameters, stage='test')
    elif configs['mode'] == 'train':
        do_train(configs=configs, parameters=parameters)
    elif configs['mode'] == 'validate':
        do_test(configs=configs, parameters=parameters, stage='validate')
    else:
        logger.error(f'There is no mode named {configs["mode"]}.')
        raise ValueError(f'There is no mode named {configs["mode"]}.')


if __name__ == '__main__':
    main()
