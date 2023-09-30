from seqeval.metrics.sequence_labeling import precision_recall_fscore_support


def get_mima_prf(labels, predictions):
    micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(
        y_true=labels
        , y_pred=predictions
        , average='micro'
    )

    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(
        y_true=labels
        , y_pred=predictions
        , average='macro'
    )

    return {
        'mip': round(number=micro_p, ndigits=3)
        , 'mir': round(number=micro_r, ndigits=3)
        , 'mif': round(number=micro_f, ndigits=3)
        , 'map': round(number=macro_p, ndigits=3)
        , 'mar': round(number=macro_r, ndigits=3)
        , 'maf': round(number=macro_f, ndigits=3)
    }
