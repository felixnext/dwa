import sys,os,argparse,time
import numpy as np
import torch
import fire
from networks.alexnet_dwa import Linear_dwa, Conv2d_dwa

import utils

tstart=time.time()

def main(seed=0, experiment='', approach='', output='', name='', nepochs=200, lr=0.05, weight_init=None, test_mode=None, log_path=None, **parameters):
    '''Trains an experiment given the current settings.

    Args:
        seed (int): Random seed
        experiment (str): Name of the experiment to load - choices: ['mnist2','pmnist','cifar','mixture']
        approach (str): Approach to take to training the experiment - choices: ['random','sgd','sgd-frozen','lwf','lfl','ewc','imm-mean','progressive','pathnet','imm-mode','sgd-restart','joint','hat','hat-test']
        output (str): Path to store the output under
        name (str): Additional experiment name for grid search
        nepochs (int): Number of epochs to iterate through
        lr (float): Learning Rate to apply 
        weight_init (str): String that defines how the weights are initialized - it can be splitted (with `:`) between convolution (first) and Linear (second) layers. Options: ["xavier", "uniform", "normal", "ones", "zeros", "kaiming"]
        test_mode (int): Defines how many tasks to iterate through
        log_path (str): Path to store detailed logs
        parameter (str): Approach dependent parameters
    '''
    # check the output path
    if output == '':
        output = '../res/' + experiment + '_' + approach + '_' + str(seed) + (("_" + name) if len(name) > 0 else "") + '.txt'
    print('=' * 100)
    print('Arguments =')
    # 
    args = {**parameters, "seed": seed, "experiment": experiment, "approach": approach, "output": output, "nepochs": nepochs, "lr": lr, "weight_init": weight_init}
    for arg in args:
        print("\t{:10}: {}".format(arg, args[arg]))
    print('=' * 100)

    ########################################################################################################################

    # Seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # check if cuda available
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    else: print('[CUDA unavailable]'); sys.exit()

    # Args -- Experiment
    if experiment=='mnist2':
        from dataloaders import mnist2 as dataloader
    elif experiment=='pmnist':
        from dataloaders import pmnist as dataloader
    elif experiment=='cifar':
        from dataloaders import cifar as dataloader
    elif experiment=='mixture':
        from dataloaders import mixture as dataloader

    # Args -- Approach
    if approach=='random':
        from approaches import random as appr
    elif approach=='sgd':
        from approaches import sgd as appr
    elif approach=='sgd-restart':
        from approaches import sgd_restart as appr
    elif approach=='sgd-frozen':
        from approaches import sgd_frozen as appr
    elif approach=='lwf':
        from approaches import lwf as appr
    elif approach=='lfl':
        from approaches import lfl as appr
    elif approach=='ewc':
        from approaches import ewc as appr
    elif approach=='imm-mean':
        from approaches import imm_mean as appr
    elif approach=='imm-mode':
        from approaches import imm_mode as appr
    elif approach=='progressive':
        from approaches import progressive as appr
    elif approach=='pathnet':
        from approaches import pathnet as appr
    elif approach=='hat-test':
        from approaches import hat_test as approach
    elif approach=='hat':
        from approaches import hat as appr
    elif approach=='joint':
        from approaches import joint as appr
    elif approach=='dwa':
        from approaches import dwa as appr

    # Args -- Network
    if experiment in ['mnist2', 'pmnist']:
        if approach in ['hat', 'hat-test']:
            from networks import mlp_hat as network
        elif approach == 'dwa':
            from networks import mlp_dwa as network
        else:
            from networks import mlp as network
    else:
        if approach=='lfl':
            from networks import alexnet_lfl as network
        elif approach=='hat':
            from networks import alexnet_hat as network
        elif approach=='progressive':
            from networks import alexnet_progressive as network
        elif approach=='pathnet':
            from networks import alexnet_pathnet as network
        elif approach=='hat-test':
            from networks import alexnet_hat_test as network
        elif approach=='dwa':
            from networks import alexnet_dwa as network
        else:
            from networks import alexnet as network

    ########################################################################################################################

    # Load
    print('Load data...')
    data,taskcla,inputsize=dataloader.get(seed=seed)
    print('Input size =',inputsize,'\nTask info =',taskcla)

    # Init the network and put on gpu
    print('Inits...')
    # handle input parameters for dwa approaches
    if approach == "dwa":
        params = {}
        for key in parameters:
            if key in ["use_processor", "processor_feats", "emb_size", "use_stem", "use_concat", "use_combination", "use_dropout"]:
                params[key] = parameters[key]
        net = network.Net(inputsize, taskcla, **params).cuda()
    else:
        net=network.Net(inputsize,taskcla).cuda()
    utils.print_model_report(net)

    # setup network weights
    if weight_init is not None:
        # retrieve init data
        inits = weight_init.split(":")
        conv_init = inits[0].split(",")
        conv_bias = conv_init[1] if len(conv_init) > 1 else "zeros"
        conv_init = conv_init[0]
        linear_init = inits[-1].split(",")
        linear_bias = linear_init[1] if len(linear_init) > 1 else "zeros"
        linear_init = linear_init[0]

        init_funcs = {
            "xavier": lambda x: torch.nn.init.xavier_uniform_(x, gain=1.0),
            "kaiming": lambda x: torch.nn.init.kaiming_normal_(x, nonlinearity="relu", mode='fan_in'),
            "normal": lambda x: torch.nn.init.normal_(x, mean=0., std=1.),
            "uniform": lambda x: torch.nn.init.uniform_(x, a=0., b=1.),
            "ones": lambda x: x.data.fill_(1.),
            "zeros": lambda x: x.data.fill_(0.)
        }

        print("Init network weights:\n\tlinear weights: {}\n\tlinear bias: {}\n\tconv weights: {}\n\tconv bias: {}".format(linear_init, linear_bias, conv_init, conv_bias))

        # setup init function
        def init_weights(m):
            if type(m) == torch.nn.Linear or type(m) == Linear_dwa:
                init_funcs[linear_init](m.weight)
                init_funcs[linear_bias](m.bias)
            if type(m) == torch.nn.Conv2d or type(m) == Conv2d_dwa:
                init_funcs[conv_init](m.weight)
                init_funcs[conv_bias](m.bias)
                
        # apply to network
        net.apply(init_weights)

    # setup the approach
    params = parameters
    if approach == 'dwa':
        params = {}
        for key in parameters:
            if key in ["sparsity", "bin_sparsity", "alpha", "delta", "lamb", "sbatch", "lr_min", "lr_factor", "lr_patience", "clipgrad", "curriculum"]:
                params[key] = parameters[key]
    appr=appr.Appr(net,nepochs=nepochs,lr=lr,log_path=log_path,**params)
    print(appr.criterion)
    utils.print_optimizer_config(appr.optimizer)
    print('-'*100)

    # Loop tasks
    acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    i = 0
    for t,ncla in taskcla:
        # check if in test mode and finish after 1 task
        i += 1
        if test_mode is not None and i > test_mode:
            print("INFO: In Test-Mode - breaking after Task {}".format(test_mode))
            break

        print('*'*100)
        print('Task {:2d} ({:s})'.format(t,data[t]['name']))
        print('*'*100)

        if approach == 'joint':
            # Get data. We do not put it to GPU
            if t==0:
                xtrain=data[t]['train']['x']
                ytrain=data[t]['train']['y']
                xvalid=data[t]['valid']['x']
                yvalid=data[t]['valid']['y']
                task_t=t*torch.ones(xtrain.size(0)).int()
                task_v=t*torch.ones(xvalid.size(0)).int()
                task=[task_t,task_v]
            else:
                xtrain=torch.cat((xtrain,data[t]['train']['x']))
                ytrain=torch.cat((ytrain,data[t]['train']['y']))
                xvalid=torch.cat((xvalid,data[t]['valid']['x']))
                yvalid=torch.cat((yvalid,data[t]['valid']['y']))
                task_t=torch.cat((task_t,t*torch.ones(data[t]['train']['y'].size(0)).int()))
                task_v=torch.cat((task_v,t*torch.ones(data[t]['valid']['y'].size(0)).int()))
                task=[task_t,task_v]
        else:
            # Get data
            xtrain=data[t]['train']['x'].cuda()
            ytrain=data[t]['train']['y'].cuda()
            xvalid=data[t]['valid']['x'].cuda()
            yvalid=data[t]['valid']['y'].cuda()
            task=t

        # Train
        appr.train(task,xtrain,ytrain,xvalid,yvalid)
        print('-'*100)

        # Free some cache
        print("INFO: Free cuda cache")
        torch.cuda.empty_cache() 

        # Test
        for u in range(t+1):
            xtest=data[u]['test']['x'].cuda()
            ytest=data[u]['test']['y'].cuda()
            test_loss,test_acc,metric_str=appr.eval(u,xtest,ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}%{} <<<'.format(u,data[u]['name'],test_loss,100*test_acc, metric_str))
            acc[t,u]=test_acc
            lss[t,u]=test_loss

        # check if result directory exists
        if not os.path.exists(os.path.dirname(output)):
            print("create output dir")
            os.makedirs(os.path.dirname(output))

        # Save
        print('Save at {}'.format(output))
        np.savetxt(output,acc,'%.4f')

    # Done
    print('*'*100)
    print('Accuracies =')
    for i in range(acc.shape[0]):
        print('\t',end='')
        for j in range(acc.shape[1]):
            print('{:5.1f}% '.format(100*acc[i,j]),end='')
        print()
    print('*'*100)
    print('Done!')

    print('[Elapsed time = {:.1f} h]'.format((time.time()-tstart)/(60*60)))

    # optionally: store logs
    if hasattr(appr, 'logs'):
        if appr.logs is not None:
            #save task names
            from copy import deepcopy
            appr.logs['task_name'] = {}
            appr.logs['test_acc'] = {}
            appr.logs['test_loss'] = {}
            for t,ncla in taskcla:
                appr.logs['task_name'][t] = deepcopy(data[t]['name'])
                appr.logs['test_acc'][t]  = deepcopy(acc[t,:])
                appr.logs['test_loss'][t]  = deepcopy(lss[t,:])
            #pickle
            import gzip
            import pickle
            with gzip.open(os.path.join(appr.logpath, os.path.basename(output) + "_logs.gzip"), 'wb') as log_file:
                pickle.dump(appr.logs, log_file, pickle.HIGHEST_PROTOCOL)

    ########################################################################################################################


if __name__ == "__main__":
    fire.Fire(main)