import sys,time
import numpy as np
import torch

import utils
from .approach import BaseApproach

########################################################################################################################

class Appr(BaseApproach):

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,curriculum=None,lamb=0.75,smax=400,thres_cosh=50,thres_emb=6):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, clipgrad,curriculum)

        print("Setting Parameters to:\n\tlamba: {}\n\tsmax: {}".format(lamb, smax))
        self.lamb=lamb          # Grid search = [0.1, 0.25, 0.5, 0.75, 1, 1.5, 2.5, 4]; chosen was 0.75
        self.smax=smax          # Grid search = [25, 50, 100, 200, 400, 800]; chosen was 400
        self.thres_cosh=thres_cosh
        self.thres_emb=thres_emb

        self.mask_pre=None
        self.mask_back=None

        return

    def _get_optimizer(self,lr=None):
        '''Generates the optimizer for the current approach.'''
        if lr is None: lr=self.lr
        return torch.optim.SGD(self.model.parameters(),lr=lr)

    def train_batch(self,t,tt,i,x,y,c,b,r):
        # retrieve relevant data
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(tt)
        
        # annealing value for the gate impact (used to reduce learning impact - i.e. reduce embedding layer plasticty over time)
        s = (self.smax - 1/self.smax)*i/len(r)+1/self.smax

        # Forward
        outputs,masks=self.model.forward(task,images,s=s)
        output=outputs[t]
        loss,_=self.hat_criterion(output,targets,masks)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Restrict layer gradients in backprop
        if t>0:
            for n,p in self.model.named_parameters():
                if n in self.mask_back:
                    p.grad.data*=self.mask_back[n]

        # Compensate embedding gradients
        for n,p in self.model.named_parameters():
            if n.startswith('e'):
                num=torch.cosh(torch.clamp(s*p.data,-self.thres_cosh,self.thres_cosh))+1
                den=torch.cosh(p.data)+1
                # update the gradient data here
                p.grad.data*=self.smax/s*num/den

        # Apply step
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
        self.optimizer.step()

        # Constrain embeddings (to avoid dead cells?)
        for n,p in self.model.named_parameters():
            if n.startswith('e'):
                p.data=torch.clamp(p.data,-self.thres_emb,self.thres_emb)

        #print(masks[-1].data.view(1,-1))
        #if i>=5*self.sbatch: sys.exit()
        #if i==0: print(masks[-2].data.view(1,-1),masks[-2].data.max(),masks[-2].data.min())

        return

    def eval_batch(self,b,t,x,y,c,items):
        if "reg" not in items:
            items["reg"] = 0
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda())

        # Forward
        outputs,masks=self.model.forward(task,images,s=self.smax)
        output=outputs[t]
        loss,reg=self.hat_criterion(output,targets,masks)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # Log
        items["loss"]+=loss.data.cpu().numpy().item()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy().item()
        items["reg"]+=reg.data.cpu().numpy().item()*len(b)

        return items

    def hat_criterion(self,outputs,targets,masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                # NOTE: all mask outputs (and therefore also mp) are sigmoid outputs
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        # NOTE: regularizaiton itself is outputted for logging purposes?
        return self.criterion(outputs,targets)+self.lamb*reg,reg

    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        # Activations mask (explicitly compute and avoid gradient)
        task=torch.autograd.Variable(torch.LongTensor([t]).cuda(),volatile=False)
        mask=self.model.mask(task,s=self.smax)
        for i in range(len(mask)):
            mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
        
        # update stored masks according to task
        if t==0:
            self.mask_pre=mask
        else:
            # combine masks based on max (to update coverage of mask)
            for i in range(len(self.mask_pre)):
                self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

        # Weights mask
        self.mask_back={}
        for n,_ in self.model.named_parameters():
            vals=self.model.get_view_for(n,self.mask_pre)
            if vals is not None:
                self.mask_back[n]=1-vals

        return

########################################################################################################################
