import os
from datetime import datetime

import pandas
import numpy
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import torchvision.datasets.folder
from torch.utils.data import TensorDataset, Subset, ConcatDataset
from torchvision.datasets import MNIST, ImageFolder
from torchvision.transforms.functional import rotate

DATASETS = [
    "EDGdroughts"
]


def get_dataset_class(dataset_name):
    """Return the dataset class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]


def num_environments(dataset_name):
    return len(get_dataset_class(dataset_name).ENVIRONMENTS)


class MultipleDomainDataset:
    N_STEPS = 5001           # Default, subclasses may override
    CHECKPOINT_FREQ = 100    # Default, subclasses may override
    N_WORKERS = 8            # Default, subclasses may override
    ENVIRONMENTS = None      # Subclasses should override
    INPUT_SHAPE = None       # Subclasses should override

    def __getitem__(self, index):
        return self.datasets[index]

    def __len__(self):
        return len(self.datasets)
    
class EDGdroughts(MultipleDomainDataset):
    def __init__(self, data_dir, test_envs, hparams):
        super().__init__()
        df = self.load_csv(data_dir)
        self.domain_dict = self.orgnize_domain_dict(df)



    def load_csv(self, dir):
        df = pandas.read_csv(dir)
        return df


    # get the domain from year
    # return { year : [ start row index , end_row index (not included)] }
    def orgnize_domain_dict(self, dataframe):
        current_year = str(dataframe['date'].iloc[0])[0:4]
        dataframe_len = len(dataframe)
        last_domain_index = 0
        for i in range(1,dataframe_len):
            this_year = str(dataframe['date'].iloc[i])[0:4]
            if this_year != current_year:
                # from last_domain_index to the i-th, i-th not included
                self.domain_dict[current_year] = [last_domain_index, i]
                last_domain_index = i
                current_year = this_year
            if i == (dataframe_len-1):
                self.domain_dict[current_year] = [last_domain_index, i+1]
        return self.domain_dict



    #
    #   return the dataset which contains the N (x, y) tensor pairs
    #   from domain I
    #   start index : the start index of domain i
    #   end index : start index + N
    #
    def orgnize_dataset(self, dataframe, start_index, end_index):
        column_index = dataframe.columns.get_loc('date')
        # start from 0
        start = 0
        current_dataset = []
        for i in range(start_index, end_index):
            tensors = []
            row = dataframe.iloc[i]
            current_flag = row['score']
            # remove label
            arr_x = row[:-1].values
            # set time from string to int in datetime
            arr_x[column_index] = datetime.fromisoformat(arr_x[column_index]).timestamp()
            # set type
            numpy_x = arr_x.astype(numpy.float64)
            x_tensor = torch.tensor(numpy_x)
            # add one dim so shape = [1, 20]
            x_tensor = x_tensor.unsqueeze(0)
            tensors.append(x_tensor)
            # concat the tensors
            if not numpy.isnan(current_flag):
                x = torch.tensor(tensors)
                arr_y = row[-1].values
                numpy_y = arr_y.astype(numpy.float64)
                y = torch.tensor(numpy_y)
                current_dataset.append((x,y))
                # at here the data should be
                # tensor x : [7, 20]
                # tensor y : [1]
                # then we can concate the whole dataset
        return current_dataset






