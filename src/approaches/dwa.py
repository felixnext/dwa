import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils
from .approach import BaseApproach

class Appr(BaseApproach):
    '''Implementation of dynamic weight allocation experiments.

    TODO: add loss formulas here to understand params
    
    Args:
        model (Net): Model to be loaded
        nepochs (int): Number of epochs to train
        sbatch (int): Batch Size
        lr (float): Learning Rate for the approach
        lr_min (float): Minimal allowed learningrate
        lr_factor (float): Factor by which the learning rate will be reduced if loss does not change
        lr_patience (int): Epochs to wait before reducing learning rate
        clipgrad (int): value at which gradients are clipped to avoid explosion
        sparsity (float): Soft-Constraint for the percentage of neurons used per layer for a specific task
        bin_sparsity (bool): Defines if sparsity should be calculated only as non-zero or gradular
        alpha (float): Weight with which the adjusted-triplet loss is taken into account
    '''

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000, sparsity=0.2, bin_sparsity=False, alpha=0.5, lamb=1, delta=1):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, clipgrad)

        # set parameters
        print("Setting Parameters to:\n\tsparsity: {}{}\n\talpha: {}\n\tdelta: {}\n\tlambda: {}".format(sparsity, " (bin)" if bin_sparsity is True else "", alpha, delta, lamb))
        self.sparsity = sparsity
        self.alpha = alpha
        self.lamb = lamb
        self.delta = delta
        self.bin_sparse = bin_sparsity

        # define constants used over training
        self.fisher = None
        self.anchor_neg = None
        self.anchor_pos = None
        self.anchor_task = None
        self.anchor_store = [None] * len(model.taskcla)

        return

    def _get_optimizer(self,lr=None):
        '''Generates the optimizer for the current approach.'''
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train_batch(self,t,i,x,y,b,r):
        # TODO: adjust
        c = 1
        # retrieve relevant data
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda())
        
        # compute forward pass
        outputs, emb, masks = self.model.forward(task, images)
        output = outputs[t]
        loss,_,_,_ = self.dwa_criterion(t, c, output, targets, emb, masks)

        # backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)   # avoid too high gradients
        self.optimizer.step()   # take an optimizer step (according to internal learning rate)
        
        return

    def eval_batch(self,b,t,x,y,items):
        # set items
        for n in ["triplet_loss", "attention_loss", "sparsity_reg"]:
            items[n] = 0
        # TODO: adjust
        c = 1
        
        # load batch
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda())
        
        # forward pass
        outputs, emb, masks = self.model.forward(task, images)
        output = outputs[t]
        loss,triplet,att,reg = self.dwa_criterion(t, c, output, targets, emb, masks)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # log relevant data
        items["loss"]+=loss.data.cpu().numpy().item()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy().item()
        items["sparsity_reg"]+=reg.data.cpu().numpy().item()*len(b)
        items["triplet_loss"]+=triplet.data.cpu().numpy().item()*len(b)
        items["attention_loss"]+=att.data.cpu().numpy().item()*len(b)
    
        return items

    def dwa_criterion(self, t, c, outputs, targets, emb, masks):
        # compute default loss
        loss = self.criterion(outputs, targets)

        # compute the triplet loss and weight on complexity
        triplet = (1-c) * utils.anchor_loss(emb, self.anchor_pos, self.anchor_neg, self.anchor_task, self.alpha, self.delta)

        # compute the remaining losses
        att = 0
        reg = 0
        for i, (mask, name) in enumerate(masks):
            # only use fisher for 
            if t > 0:
                scale = 1   # TODO: adjust scale?
                att += torch.mul(mask * scale, self.fisher[name])
            reg += utils.sparsity_regularization(mask, self.sparsity, self.bin_sparse)

        # return the combined losses
        return loss + self.alpha*triplet + self.lamb*reg, triplet, att, reg
    
    def prepare_epoch(self, t, x, y):
        # TODO: filter data on complexity (limit on 0.4)
        x_lim = None
        y_lim = None
        prev_anchors = self.anchor_store[:t]
        # search the anchors (do not explicitly set the number of searches)
        pos, neg, task_neg = utils.anchor_search(self.model, t, x_lim, y_lim, prev_anchors)

        # assign
        self.anchor_pos = pos
        self.anchor_neg = neg
        self.anchor_task = task_neg
        self.anchor_store[t] = pos      # this stores the positive anchors for each task (to use in later ones)
    
    def _fw_pass(self, model, t, b, x, y):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda())
            c = 1
        
        # compute forward pass
        outputs, emb, masks = model.forward(task, images)
        output = outputs[t]
        loss,_,_,_ = self.dwa_criterion(t, c, output, targets, emb, masks)

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
