import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils
from .approach import BaseApproach

class Appr(BaseApproach):
    """ Class implementing the Less Forgetting Learning approach described in http://arxiv.org/abs/1607.00122 """

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=200,curriculum=None,lamb=0.05):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, clipgrad,curriculum)
        self.model_old=None

        self.lamb=lamb      # Grid search = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5]; best was 0.05, but none of them really worked

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old)

        return

    def train_batch(self,t,tt,i,x,y,c,b,r):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward old model
        layer_old=None
        if t>0:
            _,layer_old=self.model_old.forward(images)

        # Forward current model
        outputs,layer=self.model.forward(images)
        output=outputs[t]
        loss=self.lfl_criterion(layer_old,layer,output,targets)

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
        layer_old=None
        if t>0:
            _,layer_old=self.model_old.forward(images)

        # Forward current model
        outputs,layer=self.model.forward(images)
        output=outputs[t]
        loss=self.lfl_criterion(layer_old,layer,output,targets)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # Log
        items["loss"]+=loss.data.cpu().numpy()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy()

        return items

    def lfl_criterion(self, layer_old, layer, output, targets):

        # Distillation loss for all previous tasks
        loss_dist = 0
        if layer_old is not None:
            loss_dist+=torch.sum((layer_old-layer).pow(2))/2

        # Cross entropy loss
        loss_ce = self.criterion(output, targets)

        return loss_ce + self.lamb*loss_dist
