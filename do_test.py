import gc
import torch

from timeit import default_timer as timer

from torch.autograd import Variable

from evaluation import get_mima_prf
from utils import convert_ids_to_tags, get_time_info_str, log


def do_test(configs, parameters, epoch=None, stage='test'):
    converters = parameters['converters']
    current_epoch = parameters['trained_epoch']
    dataloader = parameters['test_dataloader']
    device = parameters['device']
    logging_time = configs['logging_time']
    model = parameters['model']

    if epoch != None:
        current_epoch = epoch

    if stage == 'validate':
        dataloader = parameters['validation_dataloader']

    dataloader_len = len(dataloader)

    with torch.no_grad():
        model.eval()

        batch_labels = []
        batch_predictions = []
        mima_prf = None
        total_loss = 0

        start_time = timer()

        for step, data in enumerate(iterable=dataloader):
            for key in data.keys():
                data[key] = data[key].to(device)

            inputs = data['id']
            labels = Variable(data['tag'])

            outputs = model(inputs=inputs, labels=labels)

            logits = outputs.logits
            predictions = torch.argmax(input=logits, dim=2)

            one_batch_predictions, one_batch_labels = convert_ids_to_tags(
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
            total_loss += float(loss)

            if step % logging_time == 0:
                delta_time = (timer() - start_time)
                dtime_str = get_time_info_str(total_seconds=delta_time)
                rtime_str = get_time_info_str(total_seconds=\
                    delta_time*(dataloader_len-step-1)/(step+1)
                )

                temp_epoch_loss = float(total_loss / (step + 1))

                mima_prf = get_mima_prf(
                    labels=batch_labels
                    , predictions=batch_predictions
                )

                log(
                    epoch=current_epoch
                    , iterations=f'{(step+1)}/{dataloader_len}'
                    , loss=str(round(number=temp_epoch_loss, ndigits=7))
                    , lr=None
                    , other=mima_prf
                    , stage=stage
                    , time=f'{dtime_str}/{rtime_str}'
                )

        delta_time = (timer() - start_time)
        dtime_str = get_time_info_str(total_seconds=delta_time)
        rtime_str = get_time_info_str(total_seconds=0)

        temp_epoch_loss = float(total_loss / dataloader_len)

        mima_prf = get_mima_prf(
            labels=batch_labels
            , predictions=batch_predictions
        )

        log(
            epoch=current_epoch
            , iterations=f'{dataloader_len}/{dataloader_len}'
            , loss=str(round(number=temp_epoch_loss, ndigits=7))
            , lr=None
            , other=mima_prf
            , stage=stage
            , time=f'{dtime_str}/{rtime_str}'
        )

        gc.collect()
        torch.cuda.empty_cache()

        return total_loss
