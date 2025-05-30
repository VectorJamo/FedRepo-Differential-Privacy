import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders
from tools import construct_optimizer

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

class SpreadModel(nn.Module):
    def __init__(self, ws, margin):
        super().__init__()
        self.ws = nn.Parameter(ws) # Make the classifier weights as trainable parameters so that they can be optimized
        self.margin = margin # Hyperparameter that specifies that minimum value of the cosine distance that is allowed

    def forward(self):
        ws_norm = F.normalize(self.ws, dim=1) # Normalize the weights values so that the dot product of two vectors will be equal to their cosine similarity

        # Compute the cosine similarity between all pairs of normalized weight vectors using matrix multiplication.
        # Here, we subtract with 1 to make the relation between cosine similarity and dot product proportional to each other. 
        # Cosine distance = 1 - cosine similarity
        cos_dis = 0.5 * (1.0 - torch.mm(ws_norm, ws_norm.transpose(0, 1)))
        
        # Identity matrix of the same shape as the weights matrix
        d_mat = torch.diag(torch.ones(self.ws.shape[0]))
        d_mat = d_mat.to(self.ws.device)

        # Negate the identity matrix(1s replaced with 0 and 0s replaced with 1s) then multiply with the cosine distance matrix.
        # The reason we do this is that the diagonals of the cos_dis matrix is the cosine distances of two same weights, which we dont need.
        cos_dis = cos_dis * (1.0 - d_mat)

        # if (cos_dis < margin), set to replace with 1(we will to penalize this), else replace with 0
        # Now, we will have a matrix with only 1s(where we need to penalize) and 0s(Do not penalize) depending on the calculated cosine distances and the margin hyperparameter. 
        indx = ((self.margin - cos_dis) > 0.0).float()
        # Finally, Calculate the total loss. Greater the 1s in our matrix(Greater the number of weights that are close), more will be the loss.
        loss = (((self.margin - cos_dis) * indx) ** 2).mean() # (self.margin - cos_dis): Compute the margin gap(extent of penalization). This is then multiplied by
        # 'indx' matrix to filter out any weights pairs that do not need to be penalized. Now, we will be left with only the weights that need to be penalized along 
        # with the extent of penalization. Then, we compute the squard mean of all of them.
        return loss


class FedAwsDP():
    def __init__(
        self, csets, gset, model, args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())

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
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            local_models = {}

            avg_loss = Averager()
            all_per_accs = []
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
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            self.update_global(
                r=r,
                global_model=self.model,
                local_models=local_models,
            )

            # This is the only new step in the FedAwS algorithm. Once the aggregated weights have been computed, then we will perform Spreadout Regularization to maximize
            # the cosine distance between the weight vectors.
            self.update_global_classifier(
                r=r,
                model=self.model,
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

    def update_local(self, r, model, train_loader, test_loader):
        # Replace BatchNorm with GroupNorm
        model = replace_batchnorm_with_groupnorm(model)

        optimizer = construct_optimizer(
            model, self.args.lr, self.args
        )

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
            n_total_bs = max(
                int(self.args.local_epochs * len(train_loader)), 5
            )
        else:
            raise ValueError(
                "local_steps and local_epochs must not be None together"
            )

        model.train()

        loader_iter = iter(train_loader)

        avg_loss = Averager()
        per_accs = []

        for t in range(n_total_bs + 1):
            if t in [0, n_total_bs]:
                per_acc = self.test(
                    model=model,
                    loader=test_loader,
                )
                per_accs.append(per_acc)

            if t >= n_total_bs:
                break

            model.train()
            try:
                batch_x, batch_y = next(loader_iter)
            except Exception:
                loader_iter = iter(train_loader)
                batch_x, batch_y = next(loader_iter)

            if self.args.cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            hs, logits = model(batch_x)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        return model, per_accs, loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        #for name, param in global_model.state_dict().items():
            #vs = []
            #for client in local_models.keys():
                #vs.append(local_models[client].state_dict()[name])
            #vs = torch.stack(vs, dim=0)

            #try:
                #mean_value = vs.mean(dim=0)
            #except Exception:
                # for BN's cnt
                #mean_value = (1.0 * vs).mean(dim=0).long()
            #mean_state_dict[name] = mean_value
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

    # The goal of this function is to compute the Spreadout Regularization Loss.
    # It ensures that the class weight vectors (ws, corresponding to each class) in the classifier layer are well-separated in the embedding space 
    # by penalizing those that are too close.
    def update_global_classifier(self, r, model):
        ws = model.classifier.weight.data # Get the weights of the final classifier layer of the CNN
        sm = SpreadModel(ws, margin=self.args.margin)

        optimizer = torch.optim.SGD(
            sm.parameters(), lr=self.args.aws_lr, momentum=0.9
        )

        # Optimize the weights according to the loss
        for _ in range(self.args.aws_steps):
            loss = sm.forward()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print(loss.item())

        model.load_state_dict({"classifier.weight": sm.ws.data}, strict=False)

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
