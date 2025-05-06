import copy
import numpy as np

import torch
import torch.nn as nn

from utils import Averager # Simple averager
from utils import count_acc # Counts the model's accuracy
from utils import append_to_logs # For logging
from utils import format_logs # For logging

from tools import construct_dataloaders # For loading in data in batches
from tools import construct_optimizer # For specifying the optimizer and loss function

#import torchcsprng as csprng
from torch import nn, optim
 
class FedAvg():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys()) # Returns a list containing the IDs of each participating client.

        # construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            )

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
        }

    def train(self):
        # Training
        # Total number of rounds of the global model training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients)) # Get the total number of clients that will participate in 1 global training round
            # client ratio * total no of clients gives the no. of clients that will participate in one round of training and updating the global parameter values
            
            # Select 'nc' number of random clients from the 'clients' list and return their IDs
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            ) 
                
            local_models = {}

            avg_loss = Averager()
            all_per_accs = []
            for client in sam_clients:
                # Create each client instance and train them
                # Returns a client's models parameters after training, a accuracy metric, and the average training loss for that client
                local_model, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )
                
                # # lr = min(r / 10.0, 1.0) * self.args.lr
                # lr = self.args.lr
                # optimizer =optim.SGD(local_model.parameters(),lr=lr, momentum=0.9)
                # # Attach Opacus PrivacyEngine for DP
                # privacy_engine = PrivacyEngine()
                # local_model.train()
                # local_model, optimizer,train_loader= privacy_engine.make_private(
                #     module=local_model,
                #     optimizer=optimizer,
                #     data_loader=self.train_loaders[client],
                #     noise_multiplier=1.0,
                #     max_grad_norm=1.0,
                # )
            
                # Store this client's model information in a dict along with other clients
                local_models[client] = copy.deepcopy(local_model)
                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            train_loss = avg_loss.item() # Global loss(average of training loss for ALL clients)
            
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            # Update the global model's parameters
            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
            )

            if r % self.args.test_round == 0:
                # global test loader
                glo_test_acc = self.test(
                    model=self.model,
                    loader=self.glo_test_loader,
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["LOCAL_TACCS"].extend(per_accs)

                print("[R:{}] [Ls:{}] [TeAc:{}] [PAcBeg:{} PAcAft:{}]".format(
                    r, train_loss, glo_test_acc, per_accs[0], per_accs[-1]
                ))
            
        # Save the global model's state dict
        print('Saving the global models state dict.')
        torch.save(self.model.state_dict(), 'saved_models/fedavg_global_model.pth')

    # Training a single client's model
    def update_local(self, r, model, train_loader, test_loader):
        # lr = min(r / 10.0, 1.0) * self.args.lr
        lr = self.args.lr

        optimizer = construct_optimizer(
            model, lr, self.args
        )
                
        # Here, 'n_total_bs' represents the total number of batches that will be processed during the local training steps for a particular round of training(1 epoch).
        if self.args.local_steps is not None:
            n_total_bs = self.args.local_steps
        elif self.args.local_epochs is not None:
            n_total_bs = max(
                int(self.args.local_epochs * len(train_loader)), 5 # len(train_loader) = no. of batches made from the training dataset. 
                #'train_loader' contains a list of tuples (train_x, train_x) where each tuple is a batch of training data.
            )
        else:
            raise ValueError(
                "local_steps and local_epochs must not be None together"
            )

        # '.train' method does not start the training, it only changes the mode of the model to training mode.        
        model.train() 
        
    
        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]: # if t is either 0 or n_total_bs (either the first or last batch)
                per_acc = self.test( # Some accuracy related metric
                    model=model,
                    loader=test_loader,
                )
                per_accs.append(per_acc)

            if t >= n_total_bs:
                break
            # '.train' method does not start the training, it only changes the mode of the model to training mode.        
            model.train()

            # Select the next batch for next training iteration(next epoch)
            try:
                batch_x, batch_y = next(loader_iter) # ** Code modified here **
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = next(loader_iter) # ** Code modified here **

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            # Train the model. Runs the .forward() method of the model and begins training.
            #print('-'*10)
            #print(batch_x.shape)
            hs, logits = model(batch_x)
            
            # Calculate the loss for current local training epoch
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)
            
            optimizer.zero_grad()
            # Perform backpropagation to update the model parameters
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            # Keep adding the losses and calculate the average for each epoch
            avg_loss.add(loss.item())

        loss = avg_loss.item() # Get the average loss
        return model, per_accs, loss # Return the model parameters, some accuracy realted metric and the loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        for name, param in global_model.state_dict().items(): # model.state_dict() returns a dict that contains all the parameters (weights and biases) of that model.
            vs = []
            for client in local_models.keys():
                #print(type(local_models[client].state_dict()[name]))
                vs.append(local_models[client].state_dict()[name]) # Appened each client's model's parameter values inside the 'vs' list.
            vs = torch.stack(vs, dim=0) # Each element in the stack contains the model's parameter for a particular client.

            # Caluclate the weighted mean of the weights and biases from ALL the clients.
            try:
                mean_value = vs.mean(dim=0) 
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).mean(dim=0).long()
            mean_state_dict[name] = mean_value

        # Update the global model with the new averaged weights and biases.
        global_model.load_state_dict(mean_state_dict, strict=False)

    def test(self, model, loader):
        model.eval()

        acc_avg = Averager()

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, logits = model(batch_x)
                acc = count_acc(logits, batch_y)
                acc_avg.add(acc)

        acc = acc_avg.item()
        return acc

    def save_logs(self, fpath):
        all_logs_str = []
        all_logs_str.append(str(self.args))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath, all_logs_str)
