import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict

import utils
from .approach import BaseApproach

class Appr(BaseApproach):
    """ Class implementing the Incremental Moment Matching (mode) approach described in https://arxiv.org/abs/1703.08475 """

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=1000,curriculum=None,lamb=0.01):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, clipgrad,curriculum)
        self.model_old=None
        self.fisher=None

        print("Setting lambda to {}".format(lamb))
        self.lamb=lamb      # Grid search = [10, 1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]; best was 1
        #self.alpha=0.5     # We assume the same alpha for all tasks. Unrealistic to tune one alpha per task (num_task-1 alphas) when we have a lot of tasks.

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)
    
    def _fw_pass(self, model, t, b, x, y):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            target=torch.autograd.Variable(y[b])

        # Forward and backward (clear gradients and compute new ones)
        model.zero_grad()
        outputs=model.forward(images)
        loss=self.imm_criterion(t,outputs[t],target)

        return loss

    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        # Model update
        if t==0:
            self.fisher=utils.fisher_matrix_diag(t,xtrain,ytrain,self.model,self._fw_pass)
        else:
            fisher_new=utils.fisher_matrix_diag(t,xtrain,ytrain,self.model,self._fw_pass)
            for (n,p),(_,p_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                p=fisher_new[n]*p+self.fisher[n]*p_old
                self.fisher[n]+=fisher_new[n]
                p/=(self.fisher[n]==0).float()+self.fisher[n]

        # Old model save
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old)

        return

    def train_batch(self,t,i,x,y,c,b,r):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward current model
        outputs=self.model.forward(images)
        output=outputs[t]
        loss=self.imm_criterion(t,output,targets)

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

        # Forward
        outputs=self.model.forward(images)
        output=outputs[t]
        loss=self.imm_criterion(t,output,targets)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # Log
        items["loss"]+=loss.data.cpu().numpy()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy()

        return items

    def imm_criterion(self,t,output,targets):
        # L2 multiplier loss
        loss_reg=0
        if t>0:
            for p,p_old in zip(self.model.parameters(),self.model_old.parameters()):
                loss_reg+=(p-p_old).pow(2).sum()/2

        # Cross entropy loss
        loss_ce=self.criterion(output,targets)

        return loss_ce+self.lamb*loss_reg