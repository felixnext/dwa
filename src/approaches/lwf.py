import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils
from .approach import BaseApproach

class Appr(BaseApproach):
    """ Class implementing the Learning Without Forgetting approach described in https://arxiv.org/abs/1606.09282 """

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=100,curriculum=None,lamb=2,T=1):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, clipgrad,curriculum)
        self.model_old=None

        print("Setting parameters to:\n\tlamb: {}\n\tT: {}".format(lamb, T))
        self.lamb=lamb          # Grid search = [0.1, 0.5, 1, 2, 4, 8, 10]; best was 2
        self.T=T                # Grid search = [0.5,1,2,4]; best was 1

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old)

        return

    def train_batch(self,t,i,x,y,c,b,r):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward old model
        targets_old=None
        if t>0:
            targets_old=self.model_old.forward(images)

        # Forward current model
        outputs=self.model.forward(images)
        loss=self.lwf_criterion(t,targets_old,outputs,targets)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
        self.optimizer.step()

        return

    def eval_batch(self,b,t,x,y,c,items):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward old model
        targets_old=None
        if t>0:
            targets_old=self.model_old.forward(images)

        # Forward current model
        outputs=self.model.forward(images)
        loss=self.lwf_criterion(t,targets_old,outputs,targets)
        output=outputs[t]
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # Log
        items["loss"]+=loss.data.cpu().numpy()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy()

        return items

    def lwf_criterion(self,t,targets_old,outputs,targets):
        # TODO: warm-up of the new layer (paper reports that improves performance, but negligible)

        # Knowledge distillation loss for all previous tasks
        loss_dist=0
        for t_old in range(0,t):
            loss_dist+=utils.cross_entropy(outputs[t_old],targets_old[t_old],exp=1/self.T)

        # Cross entropy loss
        loss_ce=self.criterion(outputs[t],targets)

        # We could add the weight decay regularization mentioned in the paper. However, this might not be fair/comparable to other approaches

        return loss_ce+self.lamb*loss_dist
