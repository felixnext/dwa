import sys,time
import numpy as np
import torch

import utils
from .approach import BaseApproach

class Appr(BaseApproach):
    '''Implementation of dynamic weight allocation experiments.
    
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
        alpha (float): Weight with which the adjusted-triplet loss is taken into account
    '''

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000, sparsity=0.2, alpha=0.5):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, clipgrad)

        # set parameters
        self.sparsity = sparsity
        self.use_processor = use_processor
        self.emb_size = emb_size
        self.alpha = alpha

        # define constants used over training

        return

    def _get_optimizer(self,lr=None):
        '''Generates the optimizer for the current approach.'''
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train_batch(self,t,i,x,y,b,r):
        # retrieve relevant data
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
        
        # TODO: implement training

    def eval_batch(self,b,t,x,y,items):
        if "reg" not in items:
            items["reg"] = 0
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda())
        
        # TODO: implement eval

    def dwa_criterion(self, outputs, targets):
        # TODO: implement criterion
        # TODO: access fisher and embeddings (pos, neg, task_neg + store of previous negatives)
        pass

    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        # TODO: implement post train steps
        # TODO: update fisher and anchors
        pass
