import sys,time
import numpy as np
import torch
from copy import deepcopy
import torch.nn.functional as F
from collections import defaultdict

import utils
from .approach import BaseApproach

class Appr(BaseApproach):
    """ Class implementing the Incremental Moment Matching (mean) approach described in https://arxiv.org/abs/1703.08475 """

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,warmup=None,clipgrad=10000,curriculum=None,log_path=None,regularizer=0.0001,alpha=0.7):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, warmup, clipgrad,curriculum, log_path)
        self.model_old=None

        print("Setting regularizer to {}".format(regularizer))
        self.reg=regularizer    # Grid search = [0.01,0.005,0.001,0.0005,0.0001,0.00005,0.000001]; best was 0.0001
        #self.alpha=alpha       # We assume the same alpha for all tasks. Unrealistic to tune one alpha per task (num_task-1 alphas) when we have a lot of tasks.

        return

    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        # Restore best, save model as old
        if t>0:
            model_state = utils.get_model(self.model)
            model_old_state = utils.get_model(self.model_old)
            for name, param in self.model.named_parameters():
                #model_state[name]=(1-self.alpha)*model_old_state[name]+self.alpha*model_state[name]
                model_state[name]=(model_state[name]+model_old_state[name]*t)/(t+1)
            utils.set_model_(self.model,model_state)

        self.model_old=deepcopy(self.model)
        utils.freeze_model(self.model_old)
        self.model_old.eval()


        return

    def train_batch(self,t,tt,i,x,y,c,b,r):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward current model
        outputs=self.model.forward(images)
        output=outputs[t]
        loss=self.imm_criterion(output, targets, t)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
        self.optimizer.step()

        return

    def eval_batch(self,b,t,tt,x,y,c,items):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward
        outputs=self.model.forward(images)
        output=outputs[t]
        loss=self.imm_criterion(output,targets,t)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # Log
        items["loss"]+=loss.data.cpu().numpy()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy()

        return items

    def imm_criterion(self, output, targets, t):
        # L2 multiplier loss
        loss_reg=0
        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(), self.model_old.named_parameters()):
                loss_reg += torch.sum((param_old-param).pow(2))/2

        # Cross entropy loss
        loss_ce=self.criterion(output,targets)

        return loss_ce + self.reg*loss_reg
