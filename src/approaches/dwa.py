import sys,time
import numpy as np
import torch
from copy import deepcopy

# leverage tensorcores
AMP_READY = False
try:
    from apex import amp
    print("INFO: Using APEX")
    AMP_READY = True
except: pass

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
        alpha (float): Margin that is enforced by the triplet loss function
        lamb_loss (float): Scaling of the dwa dependend loss parts - Can also be a tuple or list that contains combined values in format (triplet scale, attention scale)
        lamb_reg (float): Scaling of the regularization
        delta (float): Scaling of the task dependend loss (default 1) - might be changed over tasks progression
        stiff (int): Number of epochs in which to reduce the stiffness of the aux losses to 0 (if None, ignored)
        use_anchor_first (bool): Defines if the anchor loss should be calculated for the first task
        scale_att_loss (bool): Defines if the attention loss should be scaled
        use_task_loss (bool): Defines if the task embedding loss should be used
        use_apex (bool): Defines if nvidia apex optimzation should be used if available
    '''

    def __init__(self,model,nepochs=200,sbatch=32,lr=0.075,lr_min=1e-4,lr_factor=3,lr_patience=5,warmup=[10,750],clipgrad=10000, curriculum="linear:100:0.2",log_path=None, sparsity=0.2, bin_sparsity=False, alpha=1.0, lamb_loss=[10,0.05], lamb_reg=500, delta=1, stiff=None, use_anchor_first=False, scale_att_loss=False, use_task_loss=False, use_apex=False):
        super().__init__(model, nepochs, sbatch, lr, lr_min, lr_factor, lr_patience, warmup, clipgrad, curriculum, log_path, AMP_READY and use_apex)

        # set parameters
        print("Setting Parameters to:\n\tsparsity: {}{}\n\talpha: {}\n\tdelta: {}\n\tlambda: {} / {}\n\tanchor (first task): {}\n\tscale att: {}".format(sparsity, " (bin)" if bin_sparsity is True else "", alpha, delta, lamb_loss, lamb_reg, use_anchor_first, scale_att_loss))
        self.sparsity = sparsity
        self.alpha = alpha
        if isinstance(lamb_loss, list) or isinstance(lamb_loss, tuple):
            self.lamb_loss = lamb_loss
        else:
            self.lamb_loss = (lamb_loss, lamb_loss)
        self.lamb_reg = lamb_reg
        self.delta = delta
        self.bin_sparse = utils.to_bool(bin_sparsity)
        self.use_anchor_first = utils.to_bool(use_anchor_first)
        self.scale_attention = utils.to_bool(scale_att_loss)
        self.use_task_loss = utils.to_bool(use_task_loss)
        self.stiff = stiff

        # define constants used over training
        self.fisher = None
        self.anchor_neg = None
        self.anchor_pos = None
        self.anchor_task = None
        self.anchor_store = [None] * len(model.taskcla)
        
        # some anchor settings
        self.anchor_thres = 0.4         # complexity threshold for anchor data (not use to high complexity to avoid confusion)
        self.anchor_batches = 10        # number of batches to use for anchor training
        self.max_layers = 5

        # helper to improve time on sparsity
        self.sparsity_rates = {}

        # generate all task losses
        if self.use_task_loss is True:
            print("INFO: Generating task loss")
            num_tasks = len(model.taskcla)
            emb_size = model.emb_size
            elements = emb_size / num_tasks
            ite = np.zeros((num_tasks, emb_size), np.float32)
            for i in range(num_tasks):
                ite[i, int(i*elements):int((i+1*elements))] = 1
            self.ideal_task_embs = torch.from_numpy(ite).cuda()
            self.emb_mse_criterion = torch.nn.MSELoss(reduction='none')

        return

    def train_batch(self,t,tt,i,x,y,c,b,r,e):
        # retrieve relevant data
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(tt)
            comp = torch.autograd.Variable(c[b])
        
        # compute forward pass
        stiffness = e / self.stiff if self.stiff is not None else 1.0
        outputs, emb, masks = self.model.forward(task, images)
        output = outputs[t]
        loss,_,_,_ = self.dwa_criterion(t, comp, output, targets, emb, masks, stiffness)

        # backward pass
        self.optimizer.zero_grad()

        # leverage tensorcores
        if self.use_apex is True:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)   # avoid too high gradients
        self.optimizer.step()   # take an optimizer step (according to internal learning rate)
        
        return

    def eval_batch(self,b,t,tt,x,y,c,items):
        # set items
        for n in ["triplet", "att", "sparse"]:
            if n not in items:
                items[n] = 0
        
        # load batch
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(tt)
            comp = torch.autograd.Variable(c[b])
        
        # forward pass
        outputs, emb, masks = self.model.forward(task, images)
        output = outputs[t]
        loss,triplet,att,reg = self.dwa_criterion(t, comp, output, targets, emb, masks)
        _,pred=output.max(1)
        hits=(pred==targets).float()

        # log relevant data
        items["loss"]+=loss.data.cpu().numpy().item()*len(b)
        items["acc"]+=hits.sum().data.cpu().numpy().item()
        items["sparse"]+=reg.data.cpu().numpy().item()*len(b)
        items["triplet"]+=triplet.data.cpu().numpy().item()*len(b)
        items["att"]+=att.data.cpu().numpy().item()*len(b)
    
        return items

    def dwa_criterion(self, t, c, outputs, targets, emb, masks, stiff=1.0):
        # compute default loss
        loss = self.criterion(outputs, targets)

        # compute the triplet loss and weight on complexity
        triplet = None
        if t > 0 or (self.use_anchor_first is True):
            triplet = (1-c) * utils.anchor_loss(emb, self.anchor_pos, self.anchor_neg, self.anchor_task, self.alpha, self.delta)
            triplet = triplet.sum()

        # add the embedding loss (if enabled)
        if self.model.use_processor is True and self.use_task_loss is True:
            emb_loss = (1 - c) * self.emb_mse_criterion(emb, self.ideal_task_embs[t].repeat(emb.size(0), 1)).mean(-1)
            emb_loss = emb_loss.sum()
            triplet = emb_loss if triplet is None else triplet + emb_loss

        # compute the remaining losses
        att = None
        reg = None
        for i, (mask, name) in enumerate(masks):
            # only use fisher for 
            if t > 0:
                scale = 1
                if self.scale_attention is True:
                    scale = utils.scale_attention_loss(i, self.model.use_stem, self.max_layers, start=0.2)
                # NOTE: might want to add absolute here (otherwise negative loss for tanh)
                val = torch.mul(torch.abs(mask * scale), self.fisher[name]).mean()     # tested sum - loss grows to high
                # add to loss
                att = val if att is None else att + val
            
            # compute sparsity (store rates to avoid tensor recreation)
            rate = None
            if name in self.sparsity_rates:
                rate = self.sparsity_rates[name]
            mreg, mrate = utils.sparsity_regularization(mask, self.sparsity, self.bin_sparse, rate=rate)
            if rate is None:
                self.sparsity_rates[name] = mrate
            # add to loss
            reg = mreg if reg is None else reg + mreg
        
        # return the combined losses
        loss_sum = loss
        if triplet is not None:
            loss_sum += stiff * self.lamb_loss[0] * triplet
        else:
            triplet = torch.zeros([1])
        if att is not None:
            loss_sum += stiff * self.lamb_loss[1] * att
        else:
            att = torch.zeros([1])
        if reg is not None:
            loss_sum += stiff * self.lamb_reg * reg
        else:
            reg = torch.zeros([1])
        
        return loss_sum, triplet, att, reg
    
    def prepare_epoch(self, t, x, y, c):
        # filter data on complexity (limit on 0.4)
        # FEAT: shuffle data?
        x_lim = x[c < self.anchor_thres][:self.anchor_batches*self.sbatch]
        y_lim = y[c < self.anchor_thres][:self.anchor_batches*self.sbatch]
        prev_anchors = self.anchor_store[:t]
        # search the anchors (do not explicitly set the number of searches)
        pos, neg, task_neg = utils.anchor_search(self.model, t, x_lim, y_lim, prev_anchors, self.criterion, searches=5, sbatch=self.sbatch)

        # assign
        self.anchor_pos = pos.detach()
        self.anchor_neg = neg.detach()
        if task_neg is not None:
            self.anchor_task = task_neg.detach()
        self.anchor_store[t] = pos.detach()      # this stores the positive anchors for each task (to use in later ones)
    
    def _fw_pass(self, model, t, tt, b, x, y):
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(tt)
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
    
    def introspect(self, t, x, y):
        '''Computes a forward pass on the network and returns all internal variables for introspection.'''
        # NOTE: go through x and y in one batch
        tt=torch.cuda.LongTensor([t])
        with torch.no_grad():
            images=torch.autograd.Variable(x)
            targets=torch.autograd.Variable(y)
            task=torch.autograd.Variable(tt)

        # compute the forward pass
        outputs, emb, masks = self.model.forward(task, images)
        output = outputs[t]

        # put data into array
        return {
            "images": x.data.cpu().numpy(),
            "targets": y.data.cpu().numpy(),
            "masks": dict([(name, m.data.cpu().numpy()) for m, name in masks]),
            "emb": emb.data.cpu().numpy(),
            "output": output.data.cpu().numpy()
        }
