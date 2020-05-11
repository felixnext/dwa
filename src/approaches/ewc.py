import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils
from .approach import BaseApproach

class Appr(BaseApproach):
    """ Class implementing the Elastic Weight Consolidation approach described in http://arxiv.org/abs/1612.00796 """

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=100,curriculum=None,lamb=5000):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, clipgrad,curriculum)
        self.model_old=None
        self.fisher=None

        print('Setting lambda to',lamb)
        self.lamb=lamb                      # Grid search = [500,1000,2000,5000,10000,20000,50000]; best was 5000

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr = self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train_batch(self,t,i,x,y,c,b,r):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward current model
        outputs=self.model.forward(images)
        output=outputs[t]
        # ewc criterion acts as regularization applied to all weights
        loss=self.ewc_criterion(t,output,targets)

        # Backward
        self.optimizer.zero_grad()  # clear old gradients
        loss.backward()     # compute gradients from current step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)   # avoid too high gradients
        self.optimizer.step()   # take an optimizer step (according to internal learning rate)

        return

    def eval_batch(self,b,t,x,y,c,items):
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
            # for all parameter matched over from the previous model
            for (name,param),(_,param_old) in zip(self.model.named_parameters(),self.model_old.named_parameters()):
                # add the parameter change regulation to the regularization
                loss_reg+=torch.sum(self.fisher[name]*(param_old-param).pow(2))/2   # default EWC formula

        return self.criterion(output,targets)+self.lamb*loss_reg
    
    def _fw_pass(self, model, t, b, x, y):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            target=torch.autograd.Variable(y[b])

        # Forward and backward (clear gradients and compute new ones)
        model.zero_grad()
        outputs=model.forward(images)
        loss=self.ewc_criterion(t,outputs[t],target)

        return loss

    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        # store the old model (and freeze it for gradients)
        self.model_old=deepcopy(self.model) 
        self.model_old.eval()
        utils.freeze_model(self.model_old) # Freeze the weights

        # NOTE: other option is to save models to disk and reload them after each training session (slower but more accurate?)

        # deep copy the values from the old fisher matrix (previous models)
        if t>0:
            fisher_old={}
            for n,_ in self.model.named_parameters():
                fisher_old[n]=self.fisher[n].clone()
        
        # compute the fisher matrix for the current model
        # NOTE: shouldn't it be recomputed for all outputs?
        self.fisher=utils.fisher_matrix_diag(t,xtrain,ytrain,self.model,self._fw_pass)

        # combine the fisher matrices
        if t>0:
            # Watch out! We do not want to keep t models (or fisher diagonals) in memory, therefore we have to merge fisher diagonals
            # NOTE: is that equivalent?
            for n,_ in self.model.named_parameters():
                # count the old fisher matrix t times for the number of pervious tasks
                self.fisher[n]=(self.fisher[n]+fisher_old[n]*t)/(t+1)       # Checked: it is better than the other option
                #self.fisher[n]=0.5*(self.fisher[n]+fisher_old[n])

        return
