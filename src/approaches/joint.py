import sys,time
import numpy as np
import torch
from copy import deepcopy

import utils
from .approach import BaseApproach

class Appr(BaseApproach):

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,curriculum=None):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, clipgrad,curriculum)
        self.initial_model = deepcopy(model)

        return

    def _get_optimizer(self,lr=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def prepare_train(self, t):
        self.model=deepcopy(self.initial_model) # Restart model
        return

    def train_batch(self,t,i,x,y,c,b,r):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b]).cuda()
            targets=torch.autograd.Variable(y[b]).cuda()
            tasks=torch.autograd.Variable(t[b]).cuda()

        # Forward
        outputs=self.model.forward(images)
        loss=self.criterion_train(tasks,outputs,targets)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clipgrad)
        self.optimizer.step()

        return

    def eval_validation(self,t,x,y):
        total_loss=0
        total_num=0
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            with torch.no_grad():
                images=torch.autograd.Variable(x[b]).cuda()
                targets=torch.autograd.Variable(y[b]).cuda()
                tasks=torch.autograd.Variable(t[b]).cuda()

            # Forward
            outputs=self.model.forward(images)
            loss=self.criterion_train(tasks,outputs,targets)

            # Log
            total_loss+=loss.data.cpu().numpy()*len(b)
            total_num+=len(b)

        return total_loss/total_num

    def eval_batch(self,b,t,x,y,c,items):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b]).cuda()
            targets=torch.autograd.Variable(y[b]).cuda()

        # Forward
        outputs=self.model.forward(images)
        output=outputs[t]
        loss=self.criterion(output,targets)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # Log
        items["loss"]+=loss.data.cpu().numpy()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy()

        return items

    def criterion_train(self,tasks,outputs,targets):
        loss=0
        for t in np.unique(tasks.data.cpu().numpy()):
            t=int(t)
            output=outputs[t]
            idx=(tasks==t).data.nonzero().view(-1)
            loss+=self.criterion(output[idx,:],targets[idx])*len(idx)
        return loss/targets.size(0)
