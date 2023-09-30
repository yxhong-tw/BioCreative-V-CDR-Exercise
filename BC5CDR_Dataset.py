import json

from torch.utils.data import Dataset


class BC5CDR_Dataset(Dataset):
    def __init__(self, task_name, *args, **kwargs) -> None:
        super().__init__()

        file_path = f'data/{task_name}.json'

        self.data = []

        with open(file=file_path, mode='r', encoding='UTF-8') as file:
            for one_data in file:
                self.data.append(json.loads(one_data))


    def __getitem__(self, index):
        return self.data[index]


    def __len__(self):
        return len(self.data)
