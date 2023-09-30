import gc
import torch

from timeit import default_timer as timer

from torch.autograd import Variable

from do_test import do_test
from evaluation import get_mima_prf
from utils import convert_ids_to_tags, get_time_info_str, log, save_checkpoint


def do_train(configs, parameters):
    converters = parameters['converters']
    device = parameters['device']
    # freeze_lm = configs['freeze_lm']
    logging_time = configs['logging_time']
    model = parameters['model']
    optimizer = parameters['optimizer']
    save_time = configs['save_time']
    scheduler = parameters['scheduler']
    total_epoch = configs['epoch']
    train_dataloader = parameters['train_dataloader']
    trained_epoch = parameters['trained_epoch']

    train_dataloader_len = len(train_dataloader)

    for current_epoch in range(trained_epoch+1, total_epoch):
        model.train()

        batch_labels = []
        batch_predictions = []
        epoch_loss = 0
        lr = -1
        mima_prf = None

        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        start_time = timer()

        for step, data in enumerate(iterable=train_dataloader):
            for key in data.keys():
                data[key] = data[key].to(device)

            inputs = data['id']
            labels = Variable(data['tag'])

            optimizer.zero_grad()

            outputs = model(inputs=inputs, labels=labels)

            logits = outputs.logits
            predictions = torch.argmax(input=logits, dim=2)

            one_batch_labels, one_batch_predictions = convert_ids_to_tags(
                converters=converters
                , one_batch_labels=labels.tolist()
                , one_batch_predictions=predictions.tolist()
            )

            for label, prediction in zip(
                    one_batch_labels
                    , one_batch_predictions):
                batch_labels.append(label)
                batch_predictions.append(prediction)

            loss = outputs.loss
            epoch_loss += float(loss)

            loss.backward()
            optimizer.step()

            if step % logging_time == 0:
                delta_time = (timer() - start_time)
                dtime_str = get_time_info_str(total_seconds=delta_time)
                rtime_str = get_time_info_str(total_seconds=\
                    delta_time*(train_dataloader_len-step-1)/(step+1)
                )

                temp_epoch_loss = float(epoch_loss/(step+1))
                
                mima_prf = get_mima_prf(
                    labels=batch_labels
                    , predictions=batch_predictions
                )

                log(
                    epoch=current_epoch
                    , iterations=f'{(step+1)}/{train_dataloader_len}'
                    , loss=str(round(number=temp_epoch_loss, ndigits=7))
                    , lr=str(round(number=lr, ndigits=7))
                    , other=mima_prf
                    , stage='train'
                    , time=f'{dtime_str}/{rtime_str}'
                )

        delta_time = (timer() - start_time)
        dtime_str = get_time_info_str(total_seconds=delta_time)
        rtime_str = get_time_info_str(total_seconds=0)

        temp_epoch_loss = float(epoch_loss / train_dataloader_len)

        mima_prf = get_mima_prf(
            labels=batch_labels
            , predictions=batch_predictions
        )

        log(
            epoch=current_epoch
            , iterations=f'{train_dataloader_len}/{train_dataloader_len}'
            , loss=str(round(number=temp_epoch_loss, ndigits=7))
            , lr=str(round(number=lr, ndigits=7))
            , other=mima_prf
            , stage='train'
            , time=f'{dtime_str}/{rtime_str}'
        )

        validation_loss = do_test(
            configs=configs
            , parameters=parameters
            , stage='validate'
            , epoch=current_epoch)

        scheduler.step(metrics=validation_loss)

        if (current_epoch + 1) % save_time == 0:
            save_checkpoint(
                directory_path=f'checkpoints/{configs["version"]}'
                , file_name=f'{current_epoch}.pkl'
                , model=model
                , optimizer=optimizer
                , scheduler=scheduler
                , trained_epoch=current_epoch
            )

        do_test(
            configs=configs
            , parameters=parameters
            , stage='test'
            , epoch=current_epoch
        )

        gc.collect()
        torch.cuda.empty_cache()
