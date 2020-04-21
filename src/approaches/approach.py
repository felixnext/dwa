import sys,time
import numpy as np
import torch

import utils

class BaseApproach(object):

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000):
        self.model=model

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()

        return

    def _get_optimizer(self,lr=None):
        raise NotImplementedError()

    def train(self,t,xtrain,ytrain,xvalid,yvalid):
        '''Train the network for a specific task.

        Args:
            t (int): Id of the task
            xtrain (numpy): Dataset of training images
            ytrain (numpy): Dataset of target values
            xvalid (numpy): Dataset of validation images
            yvalid (numpy): Dataset of validation targets
        '''
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                self.train_epoch(t,xtrain,ytrain)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                else:
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr)
                print()
        except KeyboardInterrupt:
            print()

        # Restore best
        utils.set_model_(self.model,best_model)

        self.post_train(t)

        return

    def train_epoch(self,t,x,y):
        self.model.train()

        r=np.arange(x.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            # TODO: check for curriculum
            x, y = self.apply_curriculum(i, x, y)
            self.train_batch(t, i, x, y, r)

        return
    
    def train_batch(self, t, i, x, y, r):
        raise NotImplementedError()
    
    def apply_curriculum(self, i, x, y):
        return x, y

    def eval(self,t,x,y):
        total_num=0
        total_items = { "loss": 0, "acc": 0 }
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]

            total_items = self.eval_epoch(b, t, x, y, total_items)
            total_num += len(b)
        
        # print everything not acc and loss
        for key in total_items:
            if key in ["acc", "loss"]:
                continue
            print('  {}:{:.3f}  '.format(key,total_items[key]/total_num),end='')

        return total_items["loss"]/total_num,total_items["acc"]/total_num
    
    def eval_epoch(self, b, t, x, y, items={}):
        # avoid gradient on this
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])

        # Forward
        outputs=self.model.forward(images)
        output=outputs[t]
        loss=self.criterion(output,targets)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # Log
        # NOTE: removed index [0] here (appears to be change in numpy) - for next 2 lines
        items["loss"] += loss.data.cpu().numpy()*len(b)
        items["acc"] += hits.sum().data.cpu().numpy()

        return items
    
    def post_train(self, t):
        raise NotImplementedError()
