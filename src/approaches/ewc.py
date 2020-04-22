import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils
from .approach import BaseApproach

class Appr(BaseApproach):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=100,lamb=5000):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, clipgrad)
        self.model_old=None
        self.fisher=None

        print('Setting lambda to',lamb)
        self.lamb=lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        # Update old
        self.model_old=deepcopy(self.model)
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights

        # Fisher ops
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        self.fisher=utils.fisher_matrix_diag(t,xtrain,ytrain,self.model,self.ewc_criterion)
        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            for n,_ in self.model.named_parameters():
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option
                #self.fisher[n]=0.5*(self.fisher[n]+fisher_old[n])

        return

    def train_batch(self,t,i,x,y,b,r):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward current model
        outputs=self.model.forward(images)
        output=outputs[t]
        loss=self.ewc_criterion(t,output,targets)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
        self.optimizer.step()

        return

    def eval_batch(self,b,t,x,y,items):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward
        outputs=self.model.forward(images)
        output=outputs[t]
        loss=self.ewc_criterion(t,output,targets)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # Log
        items["loss"]+=loss.data.cpu().numpy()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy()

        return items

    def ewc_criterion(self,t,output,targets):
        # Regularization for all previous tasks
        loss_reg=0
        if t>0:
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2

        return self.criterion(output,targets)+self.lamb*loss_reg

