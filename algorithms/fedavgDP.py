import copy
import numpy as np
import torch
import torch.nn as nn

from utils import Averager  # Averager for losses
from utils import count_acc  # Accuracy calculation
from utils import append_to_logs, format_logs  # Logging utilities
from tools import construct_dataloaders, construct_optimizer  # Dataloader and optimizer helpers
from opacus import PrivacyEngine
from opacus import GradSampleModule

# Helper function to replace BatchNorm with GroupNorm
def replace_batchnorm_with_groupnorm(model):
    """
    Replace all BatchNorm layers in the model with GroupNorm layers.
    Opacus does not support BatchNorm due to its dependence on global batch statistics.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm1d):
            setattr(model, name, nn.GroupNorm(1, module.num_features))
        elif isinstance(module, nn.BatchNorm2d):
            setattr(model, name, nn.GroupNorm(1, module.num_features))
        elif isinstance(module, nn.BatchNorm3d):
            setattr(model, name, nn.GroupNorm(1, module.num_features))
        else:
            # Recursively replace BatchNorm in child modules
            replace_batchnorm_with_groupnorm(module)
    return model

class FedAvgDP:
    def __init__(self, csets, gset, model, args):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())  # Client IDs

        # Construct dataloaders
        self.train_loaders, self.test_loaders, self.glo_test_loader = construct_dataloaders(
            self.clients, self.csets, self.gset, self.args
        )

        # Logging setup
        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
        }

    def train(self):
        # Training loop for the global model
        print("--------------------------------------------Local Model parameters for Client:")
        for name, param in self.model.named_parameters():
            print(name, param.size())

        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))  # Number of clients per round
            sam_clients = np.random.choice(self.clients, n_sam_clients, replace=False)

            local_models = {}
            avg_loss = Averager()
            all_per_accs = []

            # Train each selected client
            for client in sam_clients:
                local_model, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )
                local_models[client] = copy.deepcopy(local_model)

                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            train_loss = avg_loss.item()
            per_accs = np.array(all_per_accs).mean(axis=0)
    
            # Aggregate client models into the global model
            self.update_global(r, self.model, local_models)

            if r % self.args.test_round == 0:
                glo_test_acc = self.test(model=self.model, loader=self.glo_test_loader)

                # Add to logs
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(glo_test_acc)
                self.logs["LOCAL_TACCS"].extend(per_accs)

                print(f"[R:{r}] [Ls:{train_loss}] [TeAc:{glo_test_acc}] [PAcBeg:{per_accs[0]} PAcAft:{per_accs[-1]}]")
        
        # Save the global model's state dict
        print('Saving the global models state dict.')
        torch.save(self.model.state_dict(), 'saved_models/fedavgDP_global_model.pth')


    def update_local(self, r, model, train_loader, test_loader):
        # Replace BatchNorm with GroupNorm
        model = replace_batchnorm_with_groupnorm(model)

        # Initialize optimizer
        lr = self.args.lr
        optimizer = construct_optimizer(model, lr, self.args)

        # Attach PrivacyEngine
        privacy_engine = PrivacyEngine(secure_mode=False)
        model.train()  # Ensure the model is in training mode
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=1.0,  # Adjust as needed
            max_grad_norm=1.0,     # Gradient clipping
        )

        if self.args.local_steps is not None:
            n_total_bs = self.args.local_steps
        elif self.args.local_epochs is not None:
            n_total_bs = max(int(self.args.local_epochs * len(train_loader)), 5)
        else:
            raise ValueError("local_steps and local_epochs must not be None together")

        loader_iter = iter(train_loader)
        avg_loss = Averager()
        per_accs = []
        # Local training loop
        for t in range(n_total_bs + 1):
            # **IMPORTANT: Need to comment out this accuracy measuring code because it I think it was accessing the model's activation values before model.forward().**

            if t in [0, n_total_bs]:  # Evaluate at the beginning and end of training
                per_acc = self.test(model=model, loader=test_loader) # THIS LINE HERE PUTS THE MODEL IN .eval() MODE HENCE WE PUT THE MODEL BACK INTO TRAINING MODE LATER!
                per_accs.append(per_acc)

            if t >= n_total_bs:
                break

            try:
                batch_x, batch_y = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch_x, batch_y = next(loader_iter)

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            
            model.train()

            # Forward pass
            hs, logits = model.forward(batch_x)

            # Calculate loss
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()  # Backward pass (DP hooks will compute per-sample gradients)
            optimizer.step()
            
            avg_loss.add(loss.item())

        loss = avg_loss.item()

        # Log DP budget
        #epsilon = privacy_engine.get_epsilon(delta=1e-5)
        #print(f"Client privacy budget: ε = {epsilon:.2f}, δ = 1e-5")

        return model, per_accs, loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        # Print the global model parameters
        #print(f"--------------------------------------------Global Model parameters.")
        #for name, param in global_model.state_dict().items():
            #print(name)

        # Print the local models parameter names
        #for k, v in local_models.items():
            #print(f"--------------------------------------------Local Model parameters for Client: {k}")
            #for name, param in v.state_dict().items():
                #print(name)

        '''
        Here, the global model and all the local models have the SAME parameters. The reason we get logs saying that a parameter
        is missing in client is because the client models are wrapped in a PrivacyEngine() object. If you dont believe, un-comment 
        the above two blocks of code and then run the program.

        Local model paramters have an extra '_module.' attatched so, we added this extra line of code "name = '_module.' + name"
        '''
        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                n = '_module.' + name # <---------
                if n not in local_models[client].state_dict():
                    print(f"Key '{name}' missing in client {client}. Skipping this parameter.")
                    continue
                vs.append(local_models[client].state_dict()[n])

            if len(vs) == 0:
                print(f"No valid parameters found for key '{name}'. Skipping this parameter.")
                continue

            vs = torch.stack(vs, dim=0)
            mean_value = vs.mean(dim=0)
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def test(self, model, loader):
        model.eval()
        acc_avg = Averager()

        with torch.no_grad():
            for batch_x, batch_y in loader:
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _, logits = model(batch_x)
                acc = count_acc(logits, batch_y)
                acc_avg.add(acc)

        return acc_avg.item()

    def save_logs(self, fpath):
        all_logs_str = [str(self.args)]
        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)
        append_to_logs(fpath, all_logs_str)
