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

# https://github.com/QinbinLi/MOON

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


class MOONDP():
    def __init__(
        self,
        csets,
        gset,
        model,
        args
    ):
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())
        self.n_client = len(self.clients)

        # copy private models for each client
        self.client_models = {}
        for client in self.clients:
            self.client_models[client] = copy.deepcopy(
                model.cpu()
            )

        # to cuda
        if self.args.cuda is True:
            self.model = self.model.cuda()

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
                # to cuda
                if self.args.cuda is True:
                    self.client_models[client].cuda()

                local_model, per_accs, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    local_model=copy.deepcopy(self.client_models[client]),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                )

                local_models[client] = copy.deepcopy(local_model)

                # update local model
                self.client_models[client] = copy.deepcopy(local_model.cpu())

                avg_loss.add(loss)
                all_per_accs.append(per_accs)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

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

    def update_local(self, r, model, local_model, train_loader, test_loader):
        # Replace BatchNorm with GroupNorm
        model = replace_batchnorm_with_groupnorm(model)

        glo_model = copy.deepcopy(model)
        glo_model.eval()
        local_model.eval()

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
            hs1, _ = glo_model(batch_x)
            hs0, _ = local_model(batch_x)

            criterion = nn.CrossEntropyLoss()
            ce_loss = criterion(logits, batch_y)

            # moon loss
            ct_loss = self.contrastive_loss(
                hs, hs0.detach(), hs1.detach()
            )

            loss = ce_loss + self.args.reg_lamb * ct_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.args.max_grad_norm
            )
            optimizer.step()

            avg_loss.add(loss.item())

        loss = avg_loss.item()
        return model, per_accs, loss

    def contrastive_loss(self, hs, hs0, hs1):
        cs = nn.CosineSimilarity(dim=-1)
        sims0 = cs(hs, hs0)
        sims1 = cs(hs, hs1)

        sims = 2.0 * torch.stack([sims0, sims1], dim=1)
        labels = torch.LongTensor([1] * hs.shape[0])
        labels = labels.to(hs.device)

        criterion = nn.CrossEntropyLoss()
        ct_loss = criterion(sims, labels)
        return ct_loss

    def update_global(self, r, global_model, local_models):
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in local_models.keys():
                n = '_module.' + name # <---------
                if n not in local_models[client].state_dict():
                    print(f"Key '{name}' missing in client {client}. Skipping this parameter.")
                    continue
                vs.append(local_models[client].state_dict()[n])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.mean(dim=0)
            except Exception:
                # for BN's cnt
                vs = torch.tensor(vs)
                mean_value = (1.0 * vs).mean(dim=0).long()

            alpha = self.args.c_ratio
            mean_state_dict[name] = alpha * mean_value + (1.0 - alpha) * param

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
