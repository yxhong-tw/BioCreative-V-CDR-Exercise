import json
import logging
import random
import torch

import numpy as np

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from BC5CDR_Dataset import BC5CDR_Dataset
from BC5CDR_Formatter import BC5CDR_Formatter
from model.LM_ForTokenClassification import LM_ForTokenClassification


logger = logging.getLogger(__name__)


def initialize(configs, mode):
    initialize_seeds()

    converters = initialize_converter(configs=configs)
    device = initialize_device()
    model = initialize_model(configs=configs, device=device)
    trained_epoch = -1

    parameters = {
        'converters': converters
        , 'device': device
    }

    if mode == 'train':
        optimizer = initialize_optimizer(configs=configs, model=model)
        scheduler = initialize_scheduler(configs=configs, optimizer=optimizer)

        train_dataloader = initialize_dataloader(
            configs=configs
            , task_name='train')
        validation_dataloader = initialize_dataloader(
            configs=configs
            , task_name='validation')

    if configs['load_checkpoint'] == True:
        model, optimizer, scheduler, trained_epoch = load_checkpoint(
            configs=configs
            , model=model
            , optimizer=optimizer
            , scheduler=scheduler
            , trained_epoch=trained_epoch)

    test_dataloader = initialize_dataloader(configs=configs, task_name='test')

    parameters['model'] = model
    parameters['test_dataloader'] = test_dataloader
    parameters['trained_epoch'] = trained_epoch

    if mode == 'train':
        parameters['optimizer'] = optimizer
        parameters['scheduler'] = scheduler
        parameters['train_dataloader'] = train_dataloader
        parameters['validation_dataloader'] = validation_dataloader

    logger.info('Initialize all parameters successfully.')
    logger.info(f'Details of all parameters: \n{parameters}')

    configs_str = json.dumps(obj=configs, indent=4)
    logger.info(f'Details of all configs: \n{configs_str}')

    return parameters


def initialize_converter(configs):
    ids2tags = {}
    tags2ids = None

    with open(
            file=configs['label_file_path']
            , mode='r'
            , encoding='UTF-8') as file:
        tags2ids = json.loads(file.read())
        file.close()

    for key, value in zip(tags2ids.keys(), tags2ids.values()):
        ids2tags[value] = key

    return {
        'ids2tags': ids2tags
        , 'tags2ids': tags2ids
    }


def initialize_dataloader(configs, task_name):
    dataloader = None
    dataset_name = configs['dataset_name']

    if dataset_name == 'BC5CDR_Dataset':
        batch_size = configs['batch_size']
        collate_fn = initialize_formatter(configs=configs)
        dataset = BC5CDR_Dataset(task_name=task_name)
        shuffle = configs['dataloader_shuffle']

        dataloader = DataLoader(
            dataset=dataset
            , batch_size=batch_size
            , shuffle=shuffle
            , collate_fn=collate_fn
        )
    else:
        logger.error(f'There is no dataset named {dataset_name}.')
        raise ValueError(f'There is no dataset named {dataset_name}.')

    logger.info(f'Initializing {task_name} dataloader successfully.')

    return dataloader


def initialize_device(device_type='cuda'):
    if not torch.cuda.is_available():
        device_type = 'cpu'

    logger.info(f'Initializing device to {device_type} successfully.')

    return torch.device(device_type)


def initialize_formatter(configs):
    formatter = None
    formatter_name = configs['formatter_name']

    if formatter_name == 'BC5CDR_Formatter':
        formatter = BC5CDR_Formatter(configs=configs)
    else:
        logger.error(f'There is no formatter named {formatter_name}.')
        raise ValueError(f'There is no formatter named {formatter_name}.')

    def collate_fn(data):
        return formatter(data)

    logger.info(f'Initializing formatter to {formatter_name} successfully.')

    return collate_fn


def initialize_model(configs, device):
    model = None
    model_name = configs['model_name']

    if model_name == 'LM_ForTokenClassification':
        model = LM_ForTokenClassification(configs=configs)
    else:
        logger.error(f'There is no model named {model_name}.')
        raise ValueError(f'There is no model named {model_name}.')

    logger.info(f'Initializing model to {model_name} successfully.')

    return model.to(device)


def initialize_optimizer(configs, model):
    optimizer = None
    optimizer_name = configs['optimizer_name']

    if optimizer_name == 'Adam':
        optimizer = Adam(params=model.parameters(), lr=configs['optimizer_lr'])
    else:
        logger.error(f'There is no optimizer named {optimizer_name}.')
        raise ValueError(f'There is no optimizer named {optimizer_name}.')

    logger.info(f'Initializing optimizer to {optimizer_name} successfully.')

    return optimizer


def initialize_scheduler(configs, optimizer):
    scheduler = None
    scheduler_name = configs['scheduler_name']

    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer
            , mode=configs['scheduler_mode']
            , factor=configs['scheduler_factor']
            , patience=configs['scheduler_patience']
            , verbose=configs['scheduler_verbose']
        )
    else:
        logger.error(f'There is no scheduler named {scheduler_name}.')
        raise ValueError(f'There is no scheduler named {scheduler_name}.')

    logger.info(f'Initializing scheduler to {scheduler_name} successfully.')

    return scheduler


def initialize_seeds(seed=48763):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    logger.info(f'Initializing seed to {seed} successfully.')


def load_checkpoint(configs, model, optimizer, scheduler, trained_epoch):
    directory_path=f'checkpoints/{configs["version"]}'
    file_name=f'{configs["checkpoint_epoch"]}.pkl'

    try:
        checkpoint_parameters = torch.load(f=f'{directory_path}/{file_name}')

        model.load_state_dict(checkpoint_parameters['model'])
        optimizer.load_state_dict(checkpoint_parameters['optimizer'])
        scheduler.load_state_dict(checkpoint_parameters['scheduler'])
        trained_epoch = checkpoint_parameters['trained_epoch']
    except Exception:
        logger.error('Failed to Load checkpoint from checkpoint path.')
        raise Exception('Failed to Load checkpoint from checkpoint path.')

    logger.info('Load checkpoint from checkpoint path successfully.')

    return model, optimizer, scheduler, trained_epoch
