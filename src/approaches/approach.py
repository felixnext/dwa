import sys,time
import numpy as np
import torch

# leverage tensorcores
try:
    from apex import amp
    print("INFO: Using APEX")
except: pass
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
        warmup (int): Tuple of integers containing (NUM_EPOCH_FOR_WARMUP, DIVISOR_OF_LEARNINGRATE)
        clipgrad (int): value at which gradients are clipped to avoid explosion
        curriculum (str): String to store the curriculum information - Format: "type:epochs:start:params" - type can be ['linear', 'exp', 'log']
        log_path (Str): Path under which the log data is stored after training
    '''

    def __init__(self,model,nepochs=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,warmup=None,clipgrad=10000,curriculum=None,log_path=None, apex=False):
        self.model=model

        self.nepochs=nepochs
        self.sbatch=sbatch
        self.lr=lr
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad
        self.use_apex = apex

        self.criterion=torch.nn.CrossEntropyLoss()
        self.optimizer=self._get_optimizer()
        self._parse_curriculum(curriculum)

        # integrated logging for different values
        self.logpath = log_path
        if log_path is not None:
            print("INFO: generating log at {}".format(log_path))
            self.logs = {}
            self.logs["learning_rate"] = {}
        
        # set settings for warmup
        self.warmup = None
        if (warmup is not None) and (isinstance(warmup, tuple) or isinstance(warmup, list)):
            print("INFO: Using warmup for {} epochs with a lr divisor of {}".format(warmup[0], warmup[1]))
            self.warmup = warmup

        return
    
    def _parse_curriculum(self, val):
        '''Parses the curriculum information from format: `type:max_epochs:start:*parameters`

        Note that parameters are separated by `:` as well.
        '''
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
        '''Retrieves the optimizer (default impl)'''
        if lr is None: lr=self.lr
        optimizer = torch.optim.SGD(self.model.parameters(),lr=lr)
        if self.use_apex is True:
            self.model, optimizer = amp.initialize(self.model, optimizer, opt_level="O1")
        return optimizer

    def train(self,t,xtrain,ytrain,xvalid,yvalid):
        '''Train the network for a specific task.

        Args:
            t (int): Id of the task
            xtrain (numpy): Dataset of training images
            ytrain (numpy): Dataset of target values
            xvalid (numpy): Dataset of validation images
            yvalid (numpy): Dataset of validation targets
        '''
        # set training params
        self.prepare_train(t)
        best_loss=np.inf
        best_model=utils.get_model(self.model)
        lr=self.lr if self.warmup is None else self.lr / self.warmup[1]
        patience=self.lr_patience
        self.optimizer=self._get_optimizer(lr)
        warmup_ep = self.warmup[0] if self.warmup is not None else 0
        log_lr = []

        # compute the curriculum
        if self.curriculum is not None:
            ctrain = utils.compute_curriculum(xtrain, name="train")
            cvalid = utils.compute_curriculum(xvalid, name="test")
            cthres = self._update_threshold(0., 0, 0)
        else:
            ctrain = torch.ones(xtrain.size()[0])
            cvalid = torch.ones(xvalid.size()[0])
            cthres = 1
        # push to cpu
        ctrain = ctrain.cuda()
        cvalid = cvalid.cuda()

        # Loop epochs
        try:
            for e in range(self.nepochs):
                # Train
                clock0=time.time()
                cthres,num_used = self.train_epoch(t,xtrain,ytrain,ctrain,cthres,e)
                clock1=time.time()
                train_loss,train_acc,metric_str=self.eval(t,xtrain,ytrain,ctrain,"train")
                clock2=time.time()
                print('| Epoch {:3d}, time={:5.1f}ms/{:5.1f}ms | Cur: {:.2f} ({} of {}) | LR: {:.5f} | Train: loss={:.3f}, acc={:5.1f}%{} |'.format(e+1,
                    1000*self.sbatch*(clock1-clock0)/xtrain.size(0),1000*self.sbatch*(clock2-clock1)/xtrain.size(0),cthres,num_used,xtrain.size(0),lr,train_loss,100*train_acc,metric_str),end='')
                # Valid
                valid_loss,valid_acc,metric_str=self.eval(t,xvalid,yvalid,cvalid,"valid")
                print(' Valid: loss={:.3f}, acc={:5.1f}%{} |'.format(valid_loss,100*valid_acc,metric_str),end='')

                # log the learningrate
                log_lr.append(lr)

                # Adapt lr
                if valid_loss<best_loss:
                    best_loss=valid_loss
                    best_model=utils.get_model(self.model)
                    patience=self.lr_patience
                    print(' *',end='')
                elif cthres >= 1. and warmup_ep >= e:      # note: only break if all training data have been seen (i.e. threshold is 1)
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
                
                # check if warmup ended (and adjust optimizer)
                if (e+1) == warmup_ep:
                    lr = self.lr
                    self.optimizer=self._get_optimizer(lr)

                print()
        except KeyboardInterrupt:
            print()
        
        # add to log
        if self.logpath is not None:
            self.logs["learning_rate"][t] = np.array(log_lr)

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
        tt=torch.LongTensor([t]).cuda()

        self.prepare_epoch(t, x, y, c)

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            # retrieve batch data
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]
            self.train_batch(t, tt, i, ex, ey, ec, b,r)

            # update batchwise (filtering not implemented)
            #thres = self._update_threshold(thres, e, float(i) / len(r))
        
        # update thres (for next ep)
        if thres < 1:
            thres = self._update_threshold(thres, e + 1, 0)

        return thres, ex.size(0)
    
    def prepare_train(self, t):
        return
    
    def prepare_epoch(self, t, x, y, c):
        return
    
    def train_batch(self, t, tt, i, x, y, c, b,r):
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

    def eval(self,t,x,y,c=None,prefix="test"):
        total_num=0
        total_items = { "loss": 0, "acc": 0 }
        self.model.eval()

        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()

        # check if curriculum is given
        if c is None:
            # compute the curriculum
            if self.curriculum is not None:
                c = utils.compute_curriculum(x, name="eval")
            else:
                c = torch.ones(x.size()[0])
            c = c.cuda()

        # Loop batches
        for i in range(0,len(r),self.sbatch):
            if i+self.sbatch<=len(r): b=r[i:i+self.sbatch]
            else: b=r[i:]

            total_items = self.eval_batch(b, t, x, y, c, total_items)
            total_num += len(b)
        
        # print everything not acc and loss
        metric_str = ""
        for key in total_items:
            if key in ["acc", "loss"]:
                continue
            metric_str += ' {}: {:.3f}'.format(key,total_items[key]/total_num)
        
        # store logs
        self.store_log(total_items, prefix, t)

        return total_items["loss"]/total_num,total_items["acc"]/total_num,metric_str
    
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
    
    def store_log(self, items, prefix, t):
        # log data here (if requested)
        if self.logpath is not None:
            for metric in items:
                # check if exists
                name = "{}_{}".format(prefix, metric)
                if name not in self.logs:
                    self.logs[name] = {}
                # add values
                if t not in self.logs[name]:
                    self.logs[name][t] = np.array([items[metric]])
                else:
                    self.logs[name][t] = np.concatenate([self.logs[name][t], np.array([items[metric]])], axis=0)
    
    def post_train(self, t,xtrain,ytrain,xvalid,yvalid):
        '''Code executed after successfully training a task.'''
        return
