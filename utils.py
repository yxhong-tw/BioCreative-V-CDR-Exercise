import logging
import os
import torch

from tabulate import tabulate


logger = logging.getLogger(__name__)


def convert_ids_to_tags(converters, one_batch_labels, one_batch_predictions):
    ids2tags = converters['ids2tags']
    new_one_batch_labels = []
    new_one_batch_predictions = []

    for labels, predictions in zip(one_batch_labels, one_batch_predictions):
        new_labels = []
        new_predictions = []

        for label, prediction in zip(labels, predictions):
            if label != -100:
                new_labels.append(ids2tags[label])
                new_predictions.append(ids2tags[prediction])

        new_one_batch_labels.append(new_labels)
        new_one_batch_predictions.append(new_predictions)

    return new_one_batch_labels, new_one_batch_predictions


def get_time_info_str(total_seconds):
    total_seconds = int(total_seconds)

    hours = (total_seconds // 60 // 60)
    minutes = (total_seconds // 60 % 60)
    seconds = (total_seconds % 60)

    return ('%d:%02d:%02d' % (hours, minutes, seconds))


def initialize_logger(configs):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    logger_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    fh = logging.FileHandler(
        filename=f'logs/{configs["version"]}.log'
        , mode='a'
        , encoding='UTF-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logger_formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logger_formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def log(
        epoch
        , iterations
        , loss
        , lr
        , other
        , stage
        , time):
    header2content = {
        'epoch': epoch
        , 'stage': stage
        , 'iterations': iterations
        , 'time': time
        , 'loss': loss
        , 'lr': lr
    }

    main_headers = []
    main_contents = [[]]

    for header in header2content:
        if header2content != None:
            main_headers.append(header)
            main_contents[0].append(header2content[header])

    other_headers = None
    other_contents = None

    if isinstance(other, dict):
        other_headers = [
            'MiP', 'MiR', 'MiF'
            , 'MaP', 'MaR', 'MaF'
        ]

        other_contents = [
            [
                other['mip']
                , other['mir']
                , other['mif']
                , other['map']
                , other['mar']
                , other['maf']
            ]
        ]
    else:
        other_headers = ['Other Message']
        other_contents = [[other]]

    log_str = (
        '\n'
        + tabulate(
            tabular_data=main_contents
            , headers=main_headers
            , tablefmt='pretty'
        )
        + '\n'
        + tabulate(
            tabular_data=other_contents
            , headers=other_headers
            , tablefmt='pretty'
        )
    )

    logger.info(f'{log_str}\n')


def save_checkpoint(
        directory_path
        , file_name
        , model
        , optimizer
        , scheduler
        , trained_epoch):
    save_params = {
        'trained_epoch': trained_epoch
        , 'model': model.state_dict()
        , 'optimizer': optimizer.state_dict()
        , 'scheduler': scheduler.state_dict()
    }

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    try:
        torch.save(obj=save_params, f=f'{directory_path}/{file_name}')
    except Exception:
        logger.error(f'Failed to save model with error {Exception}.')
        raise Exception
