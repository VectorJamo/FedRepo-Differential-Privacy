# Utility imports
import os
import random
from collections import namedtuple
import numpy as np

# Pytorch
import torch

# Datasets imports
from datasets.feddata import FedData

# Fed ML Algorithms
from algorithms.fedavg import FedAvg
from algorithms.fedavgDP import FedAvgDP
from algorithms.fedreg import FedReg
from algorithms.scaffold import Scaffold
from algorithms.scaffoldDP import ScaffoldDP


from algorithms.fedopt import FedOpt
from algorithms.fedoptDP import FedOptDP
from algorithms.fednova import FedNova
from algorithms.fedaws import FedAws
from algorithms.fedawsDP import FedAwsDP
from algorithms.moon import MOON
from algorithms.feddyn import FedDyn

from algorithms.pfedme import pFedMe
from algorithms.perfedavg import PerFedAvg

from algorithms.fedrs import FedRS
from algorithms.fedphp import FedPHP
from algorithms.fedphpDP import FedPHPDP
from algorithms.scaffoldrs import ScaffoldRS

# Neural networks imports
from networks.basic_nets import get_basic_net
from networks.basic_nets import ClassifyNet

# More utility imports
from paths import save_dir
from config import default_param_dicts
from utils import setup_seed

torch.set_default_tensor_type(torch.FloatTensor)

def construct_model(args):
    try:
        input_size = args.input_size
    except Exception:
        input_size = None

    try:
        input_channel = args.input_channel
    except Exception:
        input_channel = None

    #model = get_basic_net(
        #net=args.net,
        #n_classes=args.n_classes,
        #input_size=input_size,
        #input_channel=input_channel,
    #)

    model = ClassifyNet(
        net=args.net,
        init_way="none",
        n_classes=args.n_classes
    )

    return model


def construct_algo(args):
    if args.algo == "fedavg":
        FedAlgo = FedAvg
    elif args.algo == "fedavgDP":
        FedAlgo = FedAvgDP
    elif args.algo == "fedprox":
        FedAlgo = FedReg
    elif args.algo == "fedmmd":
        FedAlgo = FedReg
    elif args.algo == "scaffold":
        FedAlgo = Scaffold
    elif args.algo == "scaffoldDP":
        FedAlgo = ScaffoldDP
    elif args.algo == "fedopt":
        FedAlgo = FedOpt
    elif args.algo == "fedoptDP":
        FedAlgo = FedOptDP
    elif args.algo == "fednova":
        FedAlgo = FedNova
    elif args.algo == "fedaws":
        FedAlgo = FedAws
    elif args.algo == "fedawsDP":
        FedAlgo = FedAwsDP
    elif args.algo == "moon":
        FedAlgo = MOON
    elif args.algo == "feddyn":
        FedAlgo = FedDyn
    elif args.algo == "pfedme":
        FedAlgo = pFedMe
    elif args.algo == "perfedavg":
        FedAlgo = PerFedAvg
    elif args.algo == "fedrs":
        FedAlgo = FedRS
    elif args.algo == "fedphp":
        FedAlgo = FedPHP
    elif args.algo == "fedphpDP":
        FedAlgo = FedPHPDP
    elif args.algo == "scaffoldrs":
        FedAlgo = ScaffoldRS
    else:
        raise ValueError("No such fed algo:{}".format(args.algo))
    return FedAlgo

# Here, cnt = the number of different settings or values for the hyperparameters that will be used. The specific values is also given where applicable.
def get_hypers(algo):
    if algo == "fedavg":
        hypers = {
            "cnt": 2,
            "none": ["none"] * 2
        }
    elif algo == "fedavgDP":
        hypers = {
            "cnt": 2,
            "none": ["none"] * 2
        }
    elif algo == "fedprox":
        hypers = {
            "cnt": 2,
            "reg_way": ["fedprox"] * 2,
            "reg_lamb": [1e-5, 1e-1]
        }
    elif algo == "fedmmd":
        hypers = {
            "cnt": 2,
            "reg_way": ["fedmmd"] * 2,
            "reg_lamb": [1e-2, 1e-3]
        }
    elif algo == "scaffold":
        hypers = {
            "cnt": 2,
            "glo_lr": [0.25, 0.5]
        }
    elif algo == "scaffoldDP":
        hypers = {
            "cnt": 2,
            "glo_lr": [0.25, 0.5]
        }
    elif algo == "fedopt":
        hypers = {
            "cnt": 2,
            "glo_optimizer": ["SGD", "Adam"],
            "glo_lr": [0.1, 3e-4],
        }
    elif algo == "fedoptDP":
        hypers = {
            "cnt": 2,
            "glo_optimizer": ["SGD", "Adam"],
            "glo_lr": [0.1, 3e-4],
        }
    elif algo == "fednova":
        hypers = {
            "cnt": 2,
            "gmf": [0.5, 0.1],
            "prox_mu": [1e-3, 1e-3],
        }
    elif algo == "fedaws":
        hypers = {
            "cnt": 2,
            "margin": [0.8, 0.5],
            "aws_steps": [30, 50],
            "aws_lr": [0.1, 0.01],
        }
    elif algo == "fedawsDP":
        hypers = {
            "cnt": 2,
            "margin": [0.8, 0.5],
            "aws_steps": [30, 50],
            "aws_lr": [0.1, 0.01],
        }
    elif algo == "moon":
        hypers = {
            "cnt": 2,
            "reg_lamb": [1e-4, 1e-2]
        }
    elif algo == "feddyn":
        hypers = {
            "cnt": 2,
            "reg_lamb": [1e-3, 1e-2]
        }
    elif algo == "pfedme":
        hypers = {
            "cnt": 2,
            "reg_lamb": [1e-4, 1e-2],
            "alpha": [0.1, 0.75],
            "k_step": [20, 10],
            "beta": [1.0, 1.0],
        }
    elif algo == "perfedavg":
        hypers = {
            "cnt": 2,
            "meta_lr": [0.05, 0.01],
        }
    elif algo == "fedrs":
        hypers = {
            "cnt": 3,
            "alpha": [0.9, 0.5, 0.1],
        }
    elif algo == "fedphp":
        hypers = {
            "cnt": 3,
            "reg_way": ["KD", "MMD", "MMD"],
            "reg_lamb": [0.05, 0.1, 0.05],
        }
    elif algo == "fedphpDP":
        hypers = {
            "cnt": 3,
            "reg_way": ["KD", "MMD", "MMD"],
            "reg_lamb": [0.05, 0.1, 0.05],
        }
    elif algo == "scaffoldrs":
        hypers = {
            "cnt": 3,
            "glo_lr": [0.5, 0.25, 0.1],
            "alpha": [0.25, 0.1, 0.5],
        }
    else:
        raise ValueError("No such fed algo:{}".format(algo))
    return hypers


def main_federated(para_dict):
    print(para_dict)
    # Basically, convert the dictonary containing meta-information into a namedtuple data type which allows us to access the value of a dictonary in object like form
    # i.e. instead of doing dict['key'] we can access it as dict.key
    param_names = para_dict.keys()
    Args = namedtuple("Args", param_names)
    args = Args(**para_dict)

    # DataSets
    try:
        n_clients = args.n_clients
    except Exception:
        n_clients = None

    try:
        nc_per_client = args.nc_per_client
    except Exception:
        nc_per_client = None

    try:
        dir_alpha = args.dir_alpha
    except Exception:
        dir_alpha = None

    # Create a class wrapper for data. This class has functionality to load, split .etc. and perform other operations on the data.
    feddata = FedData(
        dataset=args.dataset,
        split=args.split,
        n_clients=n_clients,
        nc_per_client=nc_per_client,
        dir_alpha=dir_alpha,
        n_max_sam=args.n_max_sam,
    )
    # csets = Clients data sets. Create 'n_clients' datasets with above properties. 
    # gsets = Global data set. The entire dataset.
    # csets(client datasets) is a dictonary with keys of type integer(clientID) 0 to n_clients-1 and values containing a tuple of (training_set, testing_set) for that specific client.
    # gset is the global dataset containing the entire dataset.
    csets, gset = feddata.construct() # Go inside the function to learn more

    # **** THIS DOES NOT GET EXECUTED! (For FedAvg) ******* The try block will fail (perhaps because we dont have args.dset_ratio?)
    try:
        nc = int(args.dset_ratio * len(csets)) 
        clients = list(csets.keys()) 
        sam_clients = np.random.choice(
            clients, nc, replace=False
        ) 
        csets = {
            c: info for c, info in csets.items() if c in sam_clients
        }

        n_test = int(args.dset_ratio * len(gset.xs)) 
        inds = np.random.permutation(len(gset.xs))       
        gset.xs = gset.xs[inds[0:n_test]]
        gset.ys = gset.ys[inds[0:n_test]]

    except Exception:
        pass

    feddata.print_info(csets, gset)

    # Create the initial global NN model
    model = construct_model(args)
    print(model)
    print([name for name, _ in model.named_parameters()]) # 'model.named_parameters()' returns a list of tuples containing the name of the parameters and the parameter's value itself.
    # Gives the total number of trainable parameters in the model.
    n_params = sum([
        param.numel() for param in model.parameters() # This list comprehension iterates over all the model parameters and computes the number of elements for each parameter.
    ]) 
    
    print("Total number of parameters : {}".format(n_params))

    if args.cuda:
        model = model.cuda()

    # Get the federated algorithmn depending on the meta-information specified above
    FedAlgo = construct_algo(args)
    algo = FedAlgo(
        csets=csets,
        gset=gset,
        model=model,
        args=args
    )
    algo.train()

    fpath = os.path.join(
        save_dir, args.fname
    )
    algo.save_logs(fpath)
    print(algo.logs)


def main_tumor_label(dataset, algo):
    hypers = get_hypers(algo) # Get the no. of possible values for the hyperparameters and the values themselves (where applicable) in a dictonary form.

    lr = 0.03 # Specify the learning rate.
    for net in ["TFCNN"]:
        for local_epochs in [2]: 
            for j in range(hypers["cnt"]): # iterate over the no. of possible values for the hyperparameters 
                para_dict = {} # Create an empty dict that will hold meta information about the algo used, dataset used, and other info like no. of clients, no. of rounds of training.etc
                for k, vs in default_param_dicts[dataset].items(): # Iterate over a pre-defined dictonary that contains the meta information for each dataset (click and see )
                    para_dict[k] = random.choice(vs) # Get a random value for each meta-information (if there exists multiple choices)

                # Set the algo and dataset
                para_dict["algo"] = algo
                para_dict["dataset"] = dataset
                para_dict["net"] = net
                para_dict["split"] = "label"

                # nc_per_client refers to "no of classes per client" for labeled data
                # "number of classes per client" refers to how many unique categories or classes of data each client has access to and can contribute during the training process.
                # If the dataset is for an image classification task with 10 classes (e.g., digits 0-9), and each client has access to 3 consequent classes e.g. {0, 1, 2}, 
                # then nc_per_client = 3.
                if dataset=="tumor4":
                    para_dict["nc_per_client"] = 3
                elif dataset=="tumor2":
                    para_dict["nc_per_client"] = 2

                # Specify other meta informations
                para_dict["lr"] = lr # Learning rate
                para_dict["n_clients"] = 10  # Specify the number of total clients.
                para_dict["c_ratio"] = 1.0 # c_ratio * n_clients = no. of clients that will participate in a single global training round
                para_dict["local_epochs"] = local_epochs
                para_dict["max_round"] = 500
                para_dict["test_round"] = 10

                for key, values in hypers.items():
                    if key == "cnt":
                        continue
                    else:
                        para_dict[key] = values[j]

                # Summarize the meta information
                para_dict["fname"] = "{}-K100-E{}-Label2-{}-{}.log".format(
                    dataset, local_epochs, net, lr
                )

                # Once a dictonary containing meta-information about what dataset, which algo(both fed algo and training algo(specified as net)), what hyperparameter values, 
                # learning rates, rounds, batches .etc are all specified, we move on to the next step.
                main_federated(para_dict)

# Entry point
if __name__ == "__main__":
    # set seed
    setup_seed(seed=10)

    main_tumor_label("tumor4", "fedoptDP")