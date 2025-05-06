# File created on: 2025-02-28
# Author: Suraj Neupane

import torch
from torch import nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from datasets import tumor_data

from networks import basic_nets
from config import default_param_dicts

# Base Model name
BASE_NET = 'TFCNN'
DATASET = 'tumor4'
# No. of classes in the dataset
N_CLASSES = default_param_dicts[DATASET]['n_classes'][0]

def load_model(base_net, n_classes, path):
    # Create the base model
    model = basic_nets.ClassifyNet(net=base_net, init_way='none', n_classes=n_classes)
    # Load the model
    model.load_state_dict(torch.load(path, weights_only=True))

    return model

def create_test_dataloader(dataset):
    # Load the raw data
    train_xs, train_ys, test_xs, test_ys = tumor_data.load_tumor_data(
                dataset, combine=False
            )
    # Create the test_dataset
    test_set = tumor_data.TumorDataset(
        test_xs, test_ys, is_train=False
        )
    
    # Create the test_dataloader
    test_dataloader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

    return test_dataloader
    
# Model Loading
model = load_model(base_net=BASE_NET, n_classes=4, path='saved_models/fedavg_global_model.pth')
print('Model loaded successfully!')

num_params = sum(p.numel() for p in model.parameters())
print('Number of parameters in the loaded model:', num_params)

# Data Loading
test_dataloader = create_test_dataloader(dataset=DATASET)

# Get the first batch of data
test_iterator = iter(test_dataloader)
first_batch_xs, first_batch_ys = next(test_iterator)
print(first_batch_xs.shape)
print(first_batch_ys.shape)