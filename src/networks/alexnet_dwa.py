import sys
import torch
import torch.nn.functional as F
import numpy as np
import functools
from networks.dwa_utils import Linear_dwa, Conv2d_dwa

import utils

class Net(torch.nn.Module):
    '''Alex-Net inspired variant of the dynamic weight allocation network.

    Args:
        inputsize (tuple): Input size of the images in format (channels, size, size)
        taskcla (list): List of the classes per task
        use_processor (bool): Defines if the processor should be used (otherwise embedding is one-hot task vector)
        emb_size (int): Size of the context embedding that is generated by the embedding processor (None = number of tasks)
        use_stem (int): Defines the number of stem layers should be executed first before passing to the context processor (None = use input / default = 1)
        use_concat (bool): Defines if all previous layers should be concatenated for the context processor
        use_combination (bool): Defines if the attention mask should be computed through a combination of layers to save weights
    '''

    def __init__(self,inputsize,taskcla, use_processor=True, processor_feats=(10, 32), emb_size=None, use_stem=1, use_concat=False, use_combination=True, use_dropout=False):
        super(Net, self).__init__()

        # safty checks
        if use_stem is not None and use_stem >= 5:
            raise ValueError("The value of use_stem ({}) is larger than the number of layers (5)!".format(use_stem))

        # set internal values
        ncha, size, _ = inputsize
        self.taskcla=taskcla                        # contains tasks with number of classes
        self.use_stem = use_stem                    # defines the number of stem layers to use (or None for none)
        self.use_processor = use_processor          # defines if the pre-processor should be applied (or simply use task embeddings)
        self.is_linear_processor = False if use_stem is None else use_stem > 3
        self.use_combination = use_combination      # defines if attention masks should be generated through combination (to save weights)
        self.use_concat = use_concat                # defines if input to the pre-processor should be concated
        self.use_dropout = use_dropout
        self.emb_size = len(taskcla) if emb_size is None or use_processor is False else emb_size

        # create all relevant convolutions (either native as stem or dwa masked)
        self.mask_layers = torch.nn.ModuleList()
        self.mask_shapes = []
        self.c1,s,psize1 = self._create_conv(ncha, 64,  size//8,  size, 1, use_stem, inputsize)
        self.c2,s,psize2 = self._create_conv(64,   128, size//10, s,    2, use_stem, psize1)
        self.c3,s,psize3 = self._create_conv(128,  256, 2,        s,    3, use_stem, psize2)
        self.smid=s
        self.maxpool=torch.nn.MaxPool2d(2)
        self.relu=torch.nn.ReLU()

        # check if dropout sould be added or skipped (through identity)
        if use_dropout is True:
            self.drop1=torch.nn.Dropout(0.2)
            self.drop2=torch.nn.Dropout(0.5)
        else:
            self.drop1 = torch.nn.Identity()
            self.drop2 = torch.nn.Identity()
        self.fc1,psize4 = self._create_linear(256*self.smid*self.smid, 2048, 4, use_stem, psize3)
        self.fc2,psize5 = self._create_linear(2048,                    2048, 5, use_stem, psize4)
        
        # define the names of the masks
        self.mask_names = ["c1.weight", "c2.weight", "c3.weight", "fc1.weight", "fc2.weight"]
        if use_stem is not None:
            self.mask_names = self.mask_names[use_stem:]

        # generate task processor
        # all context processor stuff should start with 'p'
        if use_processor is True:
            # params
            f_bn, f_out = processor_feats
            self.processor_size = psize5
            
            # adjust layers if input from FC
            if self.is_linear_processor:
                self.pfc1 = torch.nn.Linear(self.processor_size[0], f_bn)
                self.pfc2 = torch.nn.Linear(f_bn, f_out)
                self.pfc3 = torch.nn.Linear(f_out, self.emb_size)
                #self.pfc3 = torch.nn.Embedding(100 * len(taskcla), self.emb_size)
            else:
                # check for input size and minimize
                if self.processor_size[1] >= 14:
                    self.pc_min = torch.nn.MaxPool2d(2)
                    c, w, h = self.processor_size
                    self.processor_size = (c, w // 2, h // 2)
                else:
                    self.pc_min = torch.nn.Identity()
                
                # compute processor
                self.pc1 = torch.nn.Conv2d(self.processor_size[0], f_bn, (1,1), (1,1), 0)
                self.pc2 = torch.nn.Conv2d(f_bn, f_out, (3,3), (2,2), 1)
                cin = int(np.ceil(self.processor_size[1] / 2))
                self.pfc1 = torch.nn.Linear(cin*cin*f_out, self.emb_size)
                #self.pfc1 = torch.nn.Embedding(100 * len(taskcla), self.emb_size)

        # generate all possible heads (and put them in list - list is needed for torch to properly detect layers)
        self.last=torch.nn.ModuleList()
        for t,n in self.taskcla:
            self.last.append(torch.nn.Linear(2048,n))

        # gates for this approach
        self.gate=torch.nn.Sigmoid()

        return
    
    def _create_mask(self, out_size):
        # compute the actual size
        self.mask_shapes.append(out_size)
        flat_shape = np.prod(out_size)

        # generate the layers
        if self.use_combination is True:
            sq = np.sqrt(flat_shape)
            fac1 = functools.reduce(lambda x, y: y if flat_shape % y == 0 else (x if flat_shape % x == 0 else 1), range(1, int(sq) + 2))
            fac2 = flat_shape // fac1
            
            # generate the layers
            efc1 = torch.nn.Linear(self.emb_size, fac1)
            efc2 = torch.nn.Linear(self.emb_size, fac2)
            mod = torch.nn.ModuleList([efc1, efc2])
            self.mask_layers.append(mod)
        else:
            efc1 = torch.nn.Linear(self.emb_size, flat_shape)
            self.mask_layers.append(efc1)
        
        return
    
    def _create_conv(self, fin, fout, ksize, s, pos, stem, psize):
        '''Decides whether to create a regular or weight masked convolution.'''
        # compute new kernel size
        s=utils.compute_conv_output_size(s,ksize)
        s=s//2

        # update conv
        if stem is not None and pos <= stem:
            conv=torch.nn.Conv2d(fin,fout,kernel_size=ksize)

            if self.use_concat is True:
                psize = (psize[0] + fout, s, s)
            else:
                psize = (fout, s, s)

            return conv,s,psize
        else:
            # create the mask (computed separate)
            self._create_mask((fout, fin, ksize, ksize))
            conv=Conv2d_dwa(fin,fout,kernel_size=ksize)
            return conv,s,psize
    
    def _create_linear(self, fin, fout, pos, stem, psize):
        '''Decides whether to create a regular or weight masked linear layer.'''
        if stem is not None and pos <= stem:
            fc = torch.nn.Linear(fin,fout)

            if self.use_concat is True:
                psize = (sum(psize) + fout,)
            else:
                psize = (fout,)

            return fc, psize
        else:
            # create the mask (computed separate)
            self._create_mask((fout, fin))
            fc = Linear_dwa(fin,fout)
            return fc, psize

    def forward(self,t,x,emb=None):
        '''Computes the forward pass of the network.

        Args:
            t (int): Current task
            x (float): input images
        '''
        # define the order of the layers
        conv_list = [(self.c1, self.drop1), (self.c2, self.drop1), (self.c3, self.drop2)]
        linear_list = [(self.fc1, self.drop2), (self.fc2, self.drop2)]

        # define input for the context processor
        h = x
        p = x

        # iterate through stem layers
        if self.use_stem is not None:
            for i in range(self.use_stem):   # FIX: might use +1?
                # execute conversion to linear
                if i == len(conv_list):
                    h = h.view(x.size(0),-1)
                    # update previous values (in case of concat)
                    if self.use_concat is True:
                        p = p.view(x.size(0), -1)

                # check if linear of conv
                if i >= len(conv_list):
                    j = i - len(conv_list)
                    # linear operations (linear -> relu -> dropout)
                    h = linear_list[j][1](self.relu(linear_list[j][0](h)))
                    # concat (along last dim)
                    if self.use_concat is True:
                        p = torch.cat((p, h), -1)
                else:
                    # conv operations (conv -> relu -> dropout -> maxpooling)
                    h = self.maxpool(conv_list[i][1](self.relu(conv_list[i][0](h))))
                    # concat (along channel dim which is 1) - apply max pool to adjust size
                    if self.use_concat is True:
                        p = self.maxpool(p)
                        p = torch.cat((p, h), 1)
                
                # update in case of non-concat
                if self.use_concat is False:
                    p = h

        # generate the embedding based on input (avoid if embedding is provided)
        if emb is None:
            emb = self.processor(p) if self.use_processor else t
        elif len(emb.size()) == 1:
            emb = emb.repeat(x.size()[0], 1)

        # compute the kernel masks
        masks=self.mask(emb)

        # iterate through the remaining layers (and apply masks)
        offset = 0 if self.use_stem is None else self.use_stem
        for rid in range(len(conv_list) + len(linear_list) - offset):
            tid = rid + offset
            mask, _ = masks[rid]

            # check for conversion
            if tid == len(conv_list):
                h = h.view(x.size(0),-1)
            
            # check for linear or conv
            if tid >= len(conv_list):
                j = tid - len(conv_list)
                # linear operations (linear_dwa -> relu -> dropout)
                h = linear_list[j][1](self.relu(linear_list[j][0](h, mask)))
            else:
                # conv operations (conv_dwa -> relu -> dropout -> maxpooling)
                h = self.maxpool(conv_list[tid][1](self.relu(conv_list[tid][0](h, mask))))

        # generate list of outputs from each head
        y=[]
        for i,_ in self.taskcla:
            y.append(self.last[i](h))

        return y,emb,masks
    
    def processor(self, x):
        # compute the embedding from processor
        if self.is_linear_processor:
            emb = x.view((x.size(0), -1))
            emb = self.relu(self.pfc1(emb))
            emb = self.relu(self.pfc2(emb))
            emb = self.gate(self.pfc3(emb))
        else:
            emb = self.pc_min(x)
            emb = self.relu(self.pc1(emb))
            emb = self.relu(self.pc2(emb))
            emb = emb.view((x.size(0), -1))
            emb = self.gate(self.pfc1(emb))
        return emb

    def mask(self,emb):
        # iterate through all items
        masks = []
        for i in range(len(self.mask_layers)):
            l = self.mask_layers[i]

            # compute the mask
            if self.use_combination:
                # NOTE: axis 0 is batch_size (so we do not expand there)
                mraw = l[0](emb).unsqueeze(1) * l[1](emb).unsqueeze(-1)
            else:
                mraw = l(emb)
            
            # convert to correct shape and apply gate
            mc = self.gate(mraw.view(emb.size(0), *self.mask_shapes[i]))
            # append the mask name (for fisher id)
            masks.append((mc, self.mask_names[i]))
        return masks

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gfc1,gfc2=masks
        if n=='fc1.weight':
            post=gfc1.data.view(-1,1).expand_as(self.fc1.weight)
            pre=gc3.data.view(-1,1,1).expand((self.ec3.weight.size(1),self.smid,self.smid)).contiguous().view(1,-1).expand_as(self.fc1.weight)
            return torch.min(post,pre)
        elif n=='fc1.bias':
            return gfc1.data.view(-1)
        elif n=='fc2.weight':
            post=gfc2.data.view(-1,1).expand_as(self.fc2.weight)
            pre=gfc1.data.view(1,-1).expand_as(self.fc2.weight)
            return torch.min(post,pre)
        elif n=='fc2.bias':
            return gfc2.data.view(-1)
        elif n=='c1.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.c1.weight)
        elif n=='c1.bias':
            return gc1.data.view(-1)
        elif n=='c2.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.c2.weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.c2.weight)
            return torch.min(post,pre)
        elif n=='c2.bias':
            return gc2.data.view(-1)
        elif n=='c3.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.c3.weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.c3.weight)
            return torch.min(post,pre)
        elif n=='c3.bias':
            return gc3.data.view(-1)
        return None
