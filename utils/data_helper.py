import pickle
import torch
from torch.utils.data import Dataset, TensorDataset
from .preprocess_data import PreprocessData_Ground


class TokenDataset(Dataset):
    def __init__(self, dataset):
        self.examples = dataset

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


class DataHelper(object):
    """docstring for DataHelper"""
    def __init__(self, args, load_dataset=True):
        super(DataHelper, self).__init__()

        preprocesser = PreprocessData_Ground(args.data_dir, args.generator_type, args.context_len)
 
        self.PAD = preprocesser.PAD
        self.gpt_tokenizer = preprocesser.tokenizer

        if load_dataset:
            with open(preprocesser.ground_path, 'rb') as handle:
                dataset = pickle.load(handle)

            self.trainset = TensorDataset(dataset['train'])
            self.devset= TensorDataset(dataset['dev'])
            self.testset= TensorDataset(dataset['test'])
