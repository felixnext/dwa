import os,sys
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
from PIL import Image
# for curriculum
from skimage.filters.rank import entropy
from skimage.morphology import disk

########################################################################################################################

def print_model_report(model):
    print('-'*100)
    print(model)
    print('Dimensions =',end=' ')
    count=0
    for p in model.parameters():
        print(p.size(),end=' ')
        count+=np.prod(p.size())
    print()
    print('Num parameters = %s'%(human_format(count)))
    print('-'*100)
    return count

def human_format(num):
    magnitude=0
    while abs(num)>=1000:
        magnitude+=1
        num/=1000.0
    return '%.1f%s'%(num,['','K','M','G','T','P'][magnitude])

def print_optimizer_config(optim):
    if optim is None:
        print(optim)
    else:
        print(optim,'=',end=' ')
        opt=optim.param_groups[0]
        for n in opt.keys():
            if not n.startswith('param'):
                print(n+':',opt[n],end=', ')
        print()
    return

########################################################################################################################

def get_model(model):
    return deepcopy(model.state_dict())

def set_model_(model,state_dict):
    model.load_state_dict(deepcopy(state_dict))
    return

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
    return

########################################################################################################################

def compute_conv_output_size(Lin,kernel_size,stride=1,padding=0,dilation=1):
    '''Computes the output of a specific conv layer given the parameters and input size along one dimension.
    
    Args:
        Lin (int): Input size along the dimension
        kernel_size (int): Size of the kernel
        stride (int): Spacing between different kernel steps
        padding (int): Padding applied to the side of each image
        dilation (int): Spacing between the elements of the kernel
    
    Returns:
        Int of the size along the dimension after convolution is applied
    '''
    # note: padding increases the size of the image 
    # note: increases the effective size (as the kernel grows larger by the space between, while the stride stays the same)
    return int(np.floor(
        (Lin + 2*padding - dilation*(kernel_size-1) - 1) / float(stride) + 1
    ))

########################################################################################################################

def compute_mean_std_dataset(dataset):
    # dataset already put ToTensor
    mean=0
    std=0
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    for image, _ in loader:
        mean+=image.mean(3).mean(2)
    mean /= len(dataset)

    mean_expanded=mean.view(mean.size(0),mean.size(1),1,1).expand_as(image)
    for image, _ in loader:
        std+=(image-mean_expanded).pow(2).sum(3).sum(2)

    std=(std/(len(dataset)*image.size(2)*image.size(3)-1)).sqrt()

    return mean, std

def _resize(img, size):
  img_size = img.size
  if type(size) == tuple or type(size) == list or type(size) == np.ndarray:
    frac = min((size[0] / img_size[0], size[1] / img_size[1]))
    scale = (frac, frac)
  elif type(size) == int:
    frac = float(size) / max(img_size)
    scale = (frac, frac)
  else:
    raise ValueError("Size has unkown type ({}: {})".format(type(size), size))

  out_size = (int(img_size[0] * scale[0]), int(img_size[1] * scale[1]))
  return img.resize(out_size), scale

def _pad(img, color_arr, pos):
  if pos == "topleft":
    pad_pos = [0, 0]
  elif pos == "center":
    pad_pos = [int(np.floor((color_arr.shape[0] - img.size[0]) / 2)), int(np.floor((color_arr.shape[1] - img.size[1]) / 2))]

  color = Image.fromarray(color_arr, "RGB")
  color.paste(img, pad_pos)
  
  return color

def resize_and_pad(img, size, pad_mode, color=None):
  '''Rescales the image and pads the remaining stuff.
  Relevant pad-modes:
  - stretch = stretches the image to the new aspect ratio
  - move_center = centers the image and fills the rest with `color` (or black)
  - move_center_random = centers the image and fills the rest with random color
  - move_topleft = adds image to the top left and fills rest with color
  - move_topleft_random
  - fit_center
  - fit_center_random
  - fit_topleft
  - fit_topleft_random
  Args:
    img (numpy.array): Array containing the image data.
    size (tuple): target size of the image
    pad_mode (str): Mode used for padding the image
  Returns:
    img (numpy.array): The updated image
    scale (tuple): The stretch factors along x and y axis (for adjustment of labels)
  '''
  # simply resize the image
  if pad_mode == 'stretch':
    return img.resize(size), (size[0] / img.size[0], size[1] / img.size[1])

  # create the padding array
  # NOTE: check for data type (0 to 1 vs 0 to 255) (int vs float)
  pad_split = pad_mode.split('_')
  color_arr = None
  if len(pad_split) <= 2 or pad_split[2] != 'random':
    color = [0, 0, 0] if color is None else color
    color_arr = np.stack([np.full(size, c, dtype=np.float32) for c in color], axis=-1)
  else:
    color_arr = np.random.rand(*size, img.shape[-1] if len(img.shape) > 2 else 1)
  color_arr = (color_arr * 255.0).astype('int')

  # FEAT: add offset to the border of the image (positive and negative)

  # check if only move
  if pad_split[0] == 'move':
    # check if size is at end
    scale = [1, 1]
    if size[0] < img.size[0] or size[1] < img.size[1]:
      img, scale = _resize(img, size)
    # update the final image
    img = _pad(img, color_arr, pad_split[1])

    return img, scale

  if pad_split[0] == 'fit':
    # find most relevant side to use
    img, scale = _resize(img, size)
    # update the final image
    img = _pad(img, color_arr, pad_split[1])

    return img, scale

  raise ValueError("resize mode ({}) not found!".format(pad_split[0]))

########################################################################################################################

def fisher_matrix_diag(t,x,y,model,fw_pass,sbatch=20):
    # Init
    fisher={}
    for n,p in model.named_parameters():
        fisher[n] = torch.zeros_like(p.data)
        #fisher[n]=0*p.data      # multiply to get the right format and dtype?
    # Compute
    model.train()   # set the model in training mode
    for i in tqdm(range(0,x.size(0),sbatch),desc='Fisher diagonal',ncols=100,ascii=True):
        # go through all batches
        b=torch.LongTensor(np.arange(i,np.min([i+sbatch,x.size(0)]))).cuda()
        loss = fw_pass(model,t,b,x,y)
        loss.backward()

        # Get gradients
        for n,p in model.named_parameters():
            if p.grad is not None:
                fisher[n]+=sbatch*p.grad.data.pow(2)
    # Mean
    for n,_ in model.named_parameters():
        fisher[n]=fisher[n]/x.size(0)
        fisher[n]=torch.autograd.Variable(fisher[n],requires_grad=False)
    return fisher

########################################################################################################################

def anchor_loss(tensor, pos, neg, task_neg, alpha, delta):
    # check the shape of the tensors for stacking (bring them to batch size)
    if len(tensor.shape) > len(pos.shape):
        ones = [1] * len(pos.shape)
        pos = pos.repeat(tensor.shape[0], *ones)
        neg = neg.repeat(tensor.shape[0], *ones)
        task_neg = task_neg.repeat(tensor.shape[0], *ones)

    # calculate difference
    p  = torch.sub(tensor, pos).norm(2)
    n  = torch.sub(tensor, neg).norm(2)
    tn = torch.sub(tensor, task_neg).norm(2)

    # combine values
    # TODO: add formula
    return torch.clamp( ((delta + 1) * p) - n - (delta * tn) + alpha, min=0 )

def sparsity_regularization(mask, sparsity, binary=True):
    '''Computes the sparsity regularization as percentage of elements non-zero per element in the batch.
    
    Args:
        mask (Tensor): Attention mask tensor of shape [BATCH, WEIGHTMASK] (where WEIGHTMASK is the shape of the weight tensor)
        sparsity (float): Percentage of target sparsity (e.g. 0.2 relates to 20% target sparisty of masks)
        binary (bool): Pays only attention to non-zero elements (not gradularity)
    '''
    rank = len(mask.shape)
    dims = list(range(rank))[1:]

    # compute total attention used for each batch element (total sum of active elements per batch)
    if binary is True:
        regularization = torch.sum( (mask != 0).type(torch.float32), dim=dims)
    else:
        batch = mask.shape[0]
        abs_mask = torch.abs(mask).view(batch, -1)
        max_vals,_ = torch.max(abs_mask, dim=1)
        divs = max_vals.repeat(abs_mask.shape[1], 1).t()
        regularization = torch.sum( abs_mask / divs, dim=1)

    # ratio to total available elements
    rate = np.max((mask.shape[1:].numel(), 1.))
    regularization = torch.div(regularization, rate)

    # check against sparsity constraints and create sum
    regularization = torch.clamp(regularization - sparsity, min=0)
    #regularization = torch.sum(torch.max(regularization, dim=0))   # would only count the max activation
    regularization = torch.sum(regularization)
    return regularization

def anchor_search(model, t, x, y, prev_anchors, criterion, searches=5, sbatch=64):
    '''Searches for the anchors of the respective model using low complexity input.
    
    Args:
        t (Tensor/int): Number of the task
        x (List): Dataset with low curriculum complexity
        y (List): Targets for the dataset
        prev_anchors (List): List of tensors for the previous positive anchors for each task
        num_searches (int): Number of negative search iterations
    '''
    # search positive anchor
    vec = None

    # iterate through positive data
    r=np.arange(x.size(0))
    r=torch.LongTensor(r).cuda()
    for i in tqdm(range(0, x.size(0), sbatch), desc='positive anchor search',ncols=100,ascii=True):
        # retrieve batch data
        if i+sbatch<=len(r): b=r[i:i+sbatch]
        else: b=r[i:]
        with torch.no_grad():
            images=torch.autograd.Variable(x[b])
            targets=torch.autograd.Variable(y[b])
            task=torch.autograd.Variable(torch.LongTensor([t]).cuda())
            c = 1

        # compute the bector
        _,emb,_ = model.forward(task, images)

        # concat
        vec = emb if vec is None else torch.cat((vec, emb), axis=0)
    # calc mean
    pos = torch.mean(vec, dim=0)

    # create generator
    def ds_gen():
        # iterate the dataset
        r=np.arange(x.size(0))
        r=torch.LongTensor(r).cuda()
        for i in range(0, x.size(0), sbatch):
            # retrieve batch data
            if i+sbatch<=len(r): b=r[i:i+sbatch]
            else: b=r[i:]
            with torch.no_grad():
                images=torch.autograd.Variable(x[b])
                targets=torch.autograd.Variable(y[b])
                task=torch.autograd.Variable(torch.LongTensor([t]).cuda())
            
            yield images, targets, task

    # search negative anchor
    neg = None
    max_loss = None
    for i in tqdm(range(0,searches), desc='negative anchor search',ncols=100,ascii=True):
        # create random vector - orthogonalize to positive and normalize
        vec = torch.randn(*pos.shape)
        vec = vec - (vec.dot(pos) * pos)
        vec = vec / vec.norm(p='fro')

        # iterate through model outputs
        gen = ds_gen()
        for images, targets, task in gen():
            outputs,_,_ = model.forward(task, images, emb=vec)
            output = outputs[t]
            loss = criterion(output, targets)
        
        # update (find vector position with maximal loss)
        if max_loss is None or max_loss < loss:
            max_loss = loss
            neg = vec

    # search the task negative anchor
    task_neg = None
    max_loss = None
    for i in tqdm(range(0,len(prev_anchors)), desc='task anchor search',ncols=100,ascii=True):
        # take vector from previous tasks
        vec = prev_anchors[i]

        # iterate through model outputs
        gen = ds_gen()
        for images, targets, task in gen():
            outputs,_,_ = model.forward(task, images, emb=vec)
            output = outputs[t]
            loss = criterion(output, targets)
        
        # update (find vector position with maximal loss)
        if max_loss is None or max_loss < loss:
            max_loss = loss
            task_neg = vec
    
    return pos, neg, task_neg

########################################################################################################################

def compute_curriculum(x, wnd=5, name=None):
    '''Computes the curriculum complexity for the given dataset.'''
    # create vars
    c = torch.ones(x.size()[0])
    name = 'curriculum computation{}'.format(" ({})".format(name) if name is not None else "")

    # iterate through data
    for i in tqdm(range(0,x.size()[0]), desc=name,ncols=100,ascii=True):
        # retrieve the image
        img = x[i].mean(dim=0)
        if img.max() > 10.:
            img = img / 255.
        
        # compute complexity
        img = torch.clamp(img, min=-1., max=1.).cpu().numpy().astype("float32")
        c_img = entropy(img, disk(wnd))
        
        # set value
        c[i] = torch.pow(torch.Tensor(c_img), 2).mean()

    # normalize data
    cmin = c.min()
    cmax = c.max()
    c = (c - cmin) / torch.clamp(cmax - cmin, min=0.0001)
    return c

########################################################################################################################

def cross_entropy(outputs,targets,exp=1,size_average=True,eps=1e-5):
    out=torch.nn.functional.softmax(outputs)
    tar=torch.nn.functional.softmax(targets)
    if exp!=1:
        out=out.pow(exp)
        out=out/out.sum(1).view(-1,1).expand_as(out)
        tar=tar.pow(exp)
        tar=tar/tar.sum(1).view(-1,1).expand_as(tar)
    out=out+eps/out.size(1)
    out=out/out.sum(1).view(-1,1).expand_as(out)
    ce=-(tar*out.log()).sum(1)
    if size_average:
        ce=ce.mean()
    return ce

########################################################################################################################

def set_req_grad(layer,req_grad):
    if hasattr(layer,'weight'):
        layer.weight.requires_grad=req_grad
    if hasattr(layer,'bias'):
        layer.bias.requires_grad=req_grad
    return

########################################################################################################################

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
########################################################################################################################
