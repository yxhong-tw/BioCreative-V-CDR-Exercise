import torch.nn as nn

from transformers import AutoModelForTokenClassification


class LM_ForTokenClassification(nn.Module):
    def __init__(self, configs, *args, **kwargs) -> None:
        super().__init__()

        self.lm = AutoModelForTokenClassification.from_pretrained(
            num_labels = configs['num_labels']
            , pretrained_model_name_or_path=configs['lm_path']
        )


    def forward(self, inputs, labels=None):
        outputs = self.lm(**inputs, labels=labels)

        return outputs
