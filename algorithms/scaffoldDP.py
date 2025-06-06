import copy
import numpy as np

import torch
import torch.nn as nn

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs

from tools import construct_dataloaders

from opacus import PrivacyEngine
from opacus import GradSampleModule

# code link
# https://github.com/ramshi236/Accelerated-Federated-Learning-Over-MAC-in-Heterogeneous-Networks

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

class ScaffoldOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(
            lr=lr, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, server_control, client_control, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        ng = len(self.param_groups[0]["params"])
        names = list(server_control.keys())

        # BatchNorm: running_mean/std, num_batches_tracked
        names = [name for name in names if "running" not in name]
        names = [name for name in names if "num_batch" not in name]

        t = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                c = server_control[names[t]]
                ci = client_control[names[t]]

                # print(names[t], p.shape, c.shape, ci.shape)
                d_p = p.grad.data + c.data - ci.data
                p.data = p.data - d_p.data * group["lr"]
                t += 1
        assert t == ng
        return loss


class ScaffoldDP():
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

        # control variates
        self.server_control = self.init_control(model)
        self.set_control_cuda(self.server_control, False)

        self.client_controls = {
            client: self.init_control(model) for client in self.clients
        }

    def set_control_cuda(self, control, cuda=True):
        for name in control.keys():
            if cuda is True:
                control[name] = control[name].cuda()
            else:
                control[name] = control[name].cpu()

    def init_control(self, model):
        """ a dict type: {name: params}
        """
        control = {
            name: torch.zeros_like(
                p.data
            ).cpu() for name, p in model.state_dict().items()
        }
        return control

    def train(self):
        # Training
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.c_ratio * len(self.clients))
            sam_clients = np.random.choice(
                self.clients, n_sam_clients, replace=False
            )

            delta_models = {}
            delta_controls = {}

            avg_loss = Averager()
            all_per_accs = []
            for client in sam_clients:
                # control to gpu
                self.set_control_cuda(self.client_controls[client], False)

                # update local with control variates / ScaffoldOptimizer
                delta_model, per_accs, local_steps, loss = self.update_local(
                    r=r,
                    model=copy.deepcopy(self.model),
                    train_loader=self.train_loaders[client],
                    test_loader=self.test_loaders[client],
                    server_control=self.server_control,
                    client_control=self.client_controls[client],
                )

                client_control, delta_control = self.update_local_control(
                    delta_model=delta_model,
                    server_control=self.server_control,
                    client_control=self.client_controls[client],
                    steps=local_steps,
                    lr=self.args.lr,
                )
                self.client_controls[client] = copy.deepcopy(client_control)

                delta_models[client] = copy.deepcopy(delta_model)
                delta_controls[client] = copy.deepcopy(delta_control)
                avg_loss.add(loss)
                all_per_accs.append(per_accs)

                # control to cpu
                self.set_control_cuda(self.client_controls[client], False)

            train_loss = avg_loss.item()
            per_accs = list(np.array(all_per_accs).mean(axis=0))

            self.update_global(
                r=r,
                global_model=self.model,
                delta_models=delta_models,
            )

            new_control = self.update_global_control(
                r=r,
                control=self.server_control,
                delta_controls=delta_controls,
            )
            self.server_control = copy.deepcopy(new_control)

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

    def get_delta_model(self, model0, model1):
        """ return a dict: {name: params}
        """
        state_dict = {}
        for name, param0 in model0.state_dict().items():
            name = '_module.' + name
            param1 = model1.state_dict()[name] # CHANGED HERE.
            state_dict[name] = param0.detach() - param1.detach()
        return state_dict

    def update_local(
            self, r, model, train_loader, test_loader,
            server_control, client_control):
        
        # Replace BatchNorm with GroupNorm
        model = replace_batchnorm_with_groupnorm(model)

        # lr = min(r / 10.0, 1.0) * self.args.lr
        lr = self.args.lr

        glo_model = copy.deepcopy(model)

        # USE BASE OPTIMIZER BECAUSE OPACUS DIFFERNTIAL PRIVACY DOES NOT SUPPORT CUSTOM OPTIMIZERS.
        base_optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer = ScaffoldOptimizer(
            model.parameters(),
            lr=lr,
            weight_decay=self.args.weight_decay
        )

        # Attach PrivacyEngine
        privacy_engine = PrivacyEngine(secure_mode=False)
        model.train()  # Ensure the model is in training mode
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=base_optimizer,
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

        delta_model = self.get_delta_model(glo_model, model)

        loss = avg_loss.item()
        local_steps = n_total_bs

        return delta_model, per_accs, local_steps, loss

    def update_local_control(
            self, delta_model, server_control,
            client_control, steps, lr):
        new_control = copy.deepcopy(client_control)
        delta_control = copy.deepcopy(client_control)

        for name in delta_model.keys():
            name = name.replace('_module.', '') # CHANGED HERE
            c = server_control[name]
            ci = client_control[name]

            name = '_module.' + name
            delta = delta_model[name]

            new_ci = ci.data - c.data + delta.data / (steps * lr)
            name = name.replace('_module.', '') # CHANGED HERE
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci
        return new_control, delta_control

    def update_global(self, r, global_model, delta_models):
        state_dict = {}

        #for name, param in global_model.state_dict().items():
            #vs = []
            #for client in delta_models.keys():
                #vs.append(delta_models[client][name])
            #vs = torch.stack(vs, dim=0)

            #try:
                #mean_value = vs.mean(dim=0)
                #vs = param - self.args.glo_lr * mean_value
            #except Exception:
                # for BN's cnt
                #mean_value = (1.0 * vs).mean(dim=0).long()
                #vs = param - self.args.glo_lr * mean_value
                #vs = vs.long()

            #state_dict[name] = vs
        
        for name, param in global_model.state_dict().items():
            vs = []
            for client in delta_models.keys():
                n = '_module.' + name # <---------
                if n not in delta_models[client]: # CHANGED HERE
                    print(f"Key '{name}' missing in client {client}. Skipping this parameter.")
                    continue
                vs.append(delta_models[client][n]) # CHANGED HERE

            if len(vs) == 0:
                print(f"No valid parameters found for key '{name}'. Skipping this parameter.")
                continue

            vs = torch.stack(vs, dim=0)
            mean_value = vs.mean(dim=0)
            state_dict[name] = mean_value
        

        global_model.load_state_dict(state_dict, strict=True)

    def update_global_control(self, r, control, delta_controls):
        new_control = copy.deepcopy(control)
        for name, c in control.items():
            mean_ci = []
            for _, delta_control in delta_controls.items():
                mean_ci.append(delta_control[name])
            ci = torch.stack(mean_ci).mean(dim=0)
            new_control[name] = c - ci
        return new_control

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
