import sys,time
import numpy as np
import torch

import utils
from .approach import BaseApproach

class Appr(BaseApproach):

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,warmup=None,clipgrad=10000,curriculum=None,log_path=None):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience,warmup, clipgrad,curriculum, log_path)

        return

    def _get_optimizer(self,lr=None,warmup=None):
        if lr is None: lr=self.lr
        return torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),lr=lr)

    def prepare_train(self, t):
        #train only the column for the current task
        self.model.unfreeze_column(t)
        
    def train_batch(self, t, tt, i, x, y, c, b,r):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward
        outputs=self.model.forward(images,t)
        output=outputs[t]
        loss=self.criterion(output,targets)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
        self.optimizer.step()

        return

    def eval_batch(self, b, t, x, y, c, items={}):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward
        outputs=self.model.forward(images,t)
        output=outputs[t]
        loss=self.criterion(output,targets)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # Log
        items["loss"]+=loss.data.cpu().numpy()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy()

        return items
