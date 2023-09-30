import torch

from transformers import AutoTokenizer


class BC5CDR_Formatter:
    def __init__(self, configs, *args, **kwargs) -> None:
        super().__init__()

        self.model_name = configs['model_name']
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=configs['lm_path']
        )


    def __call__(self, data):
        batch_ids = []
        batch_tags = []
        batch_tokens = []

        for one_data in data:
            batch_tags.append(one_data['tags'])
            batch_tokens.append(one_data['tokens'])

        batch_ids = self.tokens2ids(batch_tokens=batch_tokens)
        batch_tags = self.align_tags(batch_ids=batch_ids, batch_tags=batch_tags)

        return {
            'id': batch_ids
            , 'tag': torch.tensor(batch_tags)
        }


    def tokens2ids(self, batch_tokens):
        batch_ids = self.tokenizer(
            text=batch_tokens
            # 'max_length' will increase GPU memory usage significantly.
            # Instead 'max_length' with "True" to save computational resource.
            # , padding='max_length'
            , add_special_tokens=True
            , padding=True
            , truncation=True
            , max_length=512
            , is_split_into_words=True
            , return_tensors='pt'
        )

        return batch_ids


    def align_tags(self, batch_ids, batch_tags):
        new_batch_tags = []

        for batch_index, tags in enumerate(batch_tags):
            new_tags = []
            word_ids = batch_ids.word_ids(batch_index=batch_index)

            for word_id in word_ids:
                if word_id == None:
                    # TODO: Why only "-100" can work?
                    new_tags.append(-100)
                else:
                    new_tags.append(tags[word_id])

            new_batch_tags.append(new_tags)

        return new_batch_tags
