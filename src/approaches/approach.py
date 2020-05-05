import sys,time
import numpy as np
import torch

import utils

class BaseApproach(object):
    '''Abstract Baseclass for an approach toward multi-task learning

    Args:
        model (Net): Model to be loaded
        nepochs (int): Number of epochs to train
        sbatch (int): Batch Size
        lr (float): Learning Rate for the approach
        lr_min (float): Minimal allowed learningrate
        lr_factor (float): Factor by which the learning rate will be reduced if loss does not change
        lr_patience (int): Epochs to wait before reducing learning rate
        clipgrad (int): value at which gradients are clipped to avoid explosion
        curriculum (str): String to store the curriculum information - Format: "type:epochs:start:params" - type can be ['linear', 'exp', 'log']
    '''

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,curriculum=None):
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
        self._parse_curriculum(curriculum)

        return
    
    def _parse_curriculum(self, val):
        if val is None:
            self.curriculum = None
            return

        vals = str.split(val, ":")
        
        if len(vals) < 2:
            raise ValueError("Expected curriculum value to have at least 2 values")
        
        self.curriculum = {
            "type": vals[0],
            "epochs": int(vals[1]),
            "start": float(vals[2]) if len(vals) > 2 else 0.2,
            "params": vals[3:]
        }

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

        # compute the curriculum
        if self.curriculum is not None:
            # TODO: allow to adjust params
            ctrain = utils.compute_curriculum(xtrain, name="train")
            cvalid = utils.compute_curriculum(xvalid, name="test")
            cthres = self._update_threshold(0., 0, 0)
        else:
            ctrain = torch.ones(xtrain.size()[0])
            cvalid = torch.ones(xvalid.size()[0])
            cthres = 1

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                cthres = self.train_epoch(t,xtrain,ytrain,ctrain,cthres,e)
                clock1=time.time()
                train_loss,train_acc=self.eval(t,xtrain,ytrain,ctrain)
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Cur: {:.2f} | Train: loss={:.3f}, acc={:5.1f}% |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),cthres,train_loss,100*train_acc),end='')
                # Valid
                valid_loss,valid_acc=self.eval(t,xvalid,yvalid,cvalid)
                print(' Valid: loss={:.3f}, acc={:5.1f}% |'.format(valid_loss,100*valid_acc),end='')
                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                elif cthres >= 1.:      # note: only break if all training data have been seen (i.e. threshold is 1)
                    patience-=1
                    if patience<=0:
                        lr/=self.lr_factor
                        print(' lr={:.1e}'.format(lr),end='')
                        if lr<self.lr_min:
                            print()
                            break
                        patience=self.lr_patience
                        self.optimizer=self._get_optimizer(lr)
                else:
                    patience=self.lr_patience
                print()
        except KeyboardInterrupt:
            print()

        # Restore best
        utils.set_model_(self.model,best_model)

        self.post_train(t,xtrain,ytrain,xvalid,yvalid)

        return
    
    def _update_threshold(self, thres, e, b):
        # batch should be given as perc of total
        x = e + b

        # check if enabled
        if self.curriculum is None:
            return 1.

        # iterate through options
        ctype = self.curriculum["type"]
        s = self.curriculum["start"]
        me = self.curriculum["epochs"]
        if ctype == "linear":
            thres = s + (((1 - s) / me) * x)
        elif ctype == "exp":
            p = float(self.curriculum["params"][0]) if len(self.curriculum["params"]) > 0 else 2
            c = (1-s) / np.power(me, p)
            thres = s + (c*np.power(x, p))
        elif ctype == "log":
            c = (1-s) / np.log(me)
            thres = s + (c*np.log(x))
        else:
            raise ValueError("Unkown curriculum function {}".format(ctype))

        # ensure bounds
        return min(thres, 1.)

    def train_epoch(self,t,x,y,c,thres,e):
        self.model.train()

        # filter train data for the current epoch
        ex, ey, ec = self.apply_curriculum(t, x, y, c,thres)
        r=np.arange(ex.size(0))
        np.random.shuffle(r)
        r=torch.LongTensor(r).cuda()

        self.prepare_epoch(t, x, y, c)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            # TODO: apply per batch filtering on curric?
            # retrieve batch data
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            self.train_batch(t, i, ex, ey, ec, b,r)

            # update batchwise (filtering not implemented)
            #thres = self._update_threshold(thres, e, float(i) / len(r))
        
        # update thres (for next ep)
        if thres < 1:
            thres = self._update_threshold(thres, e + 1, 0)

        return thres
    
    def prepare_epoch(self, t, x, y, c):
        return
    
    def train_batch(self, t, i, x, y, c, b,r):
        '''Code to train a single batch.'''
        raise NotImplementedError()
    
    def apply_curriculum(self, t, x, y, c, thres):
        '''Code to apply curriculum learning (if enabled).
        
        Return:
            x (dataset): Train images
            y (dataset): Targets
            c (dataset): Complexity for each image
            thres (float): Threshold of the complexity
        '''
        idx = c <= thres
        return x[idx], y[idx], c[idx]

    def eval(self,t,x,y,c):
        total_num=0
        total_items = { "loss": 0, "acc": 0 }
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]

            total_items = self.eval_batch(b, t, x, y, c, total_items)
            total_num += len(b)
        
        # print everything not acc and loss
        for key in total_items:
            if key in ["acc", "loss"]:
                continue
            print('  {}:{:.3f}  '.format(key,total_items[key]/total_num),end='')

        return total_items["loss"]/total_num,total_items["acc"]/total_num
    
    def eval_batch(self, b, t, x, y, c, items={}):
        '''Eval code for a single batch.'''
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
    
    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        '''Code executed after successfully training a task.'''
        return
