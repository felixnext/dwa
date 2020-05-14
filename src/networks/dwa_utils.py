import torch
import torch.nn.functional as F

class Conv2d_dwa(torch.nn.Module):
    '''Conv2d Layer that implements dynamic weight mask.
    
    Note: this layer is not batched and might be slow - potential for optimization
    '''
    def __init__(self, fin, fout, kernel_size, stride=(1,1), padding=(0,0), dilation=(1,1)):
        super().__init__()
        # create the weights
        self.weight = torch.nn.Parameter(torch.Tensor(fout, fin, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.Tensor(fout))
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # store sizes
        self._fout = fout
        self._fin = fin
        self._kw = kernel_size
        self._kh = kernel_size
    
    def forward(self, x, m):
        mask_weight = torch.mul(self.weight, m)
        b_size = x.size()[0]
        #out = []

        # compute conv for each element in batch
        out = F.conv2d(x.view(1, b_size*self._fin, x.size()[2], x.size()[3]), mask_weight.view(b_size*self._fout, self._fin, self._kh, self._kw), 
            self.bias.repeat(b_size), self.stride, self.padding, self.dilation, groups=b_size)
        out = out.view(b_size, self._fout, out.size()[2], out.size()[3])
        #for i in range(x.size()[0]):
        #    out.append(F.conv2d(x[i:i+1], mask_weight[i, ...], self.bias, self.stride, self.padding, self.dilation))
        #out = torch.cat(out, dim=0)
        return out

class Linear_dwa(torch.nn.Module):
    '''Dense Layer that implements dynamic weight mask.'''
    def __init__(self, fin, fout):
        super().__init__()
        # create the weights
        self.weight = torch.nn.Parameter(torch.Tensor(fout, fin))
        self.bias = torch.nn.Parameter(torch.Tensor(fout))
    
    def forward(self, x, m):
        # expand the weights for the relevant process
        mask_weight = torch.mul(self.weight, m).permute(0,2,1)
        out = torch.einsum("ac,acb->ab", x, mask_weight) + self.bias
        return out

