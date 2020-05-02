import sys,os,argparse,time
import numpy as np
import torch
import fire

import utils

tstart=time.time()

def main(seed=0, experiment='', approach='', output='', nepochs=200, lr=0.05, **parameters):
    '''Trains an experiment given the current settings.

    Args:
        seed (int): Random seed
        experiment (str): Name of the experiment to load - choices: ['mnist2','pmnist','cifar','mixture']
        approach (str): Approach to take to training the experiment - choices: ['random','sgd','sgd-frozen','lwf','lfl','ewc','imm-mean','progressive','pathnet','imm-mode','sgd-restart','joint','hat','hat-test']
        output (str): Path to store the output under
        nepochs (int): Number of epochs to iterate through
        lr (float): Learning Rate to apply 
        parameter (str): Approach dependent parameters
    '''
    # check the output path
    if output == '':
        output = '../res/' + experiment + '_' + approach + '_' + str(seed) + '.txt'
    print('=' * 100)
    print('Arguments =')
    # 
    args = {**parameters, "seed": seed, "experiment": experiment, "approach": approach, "output": output, "nepochs": nepochs, "lr": lr}
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
            # TODO: implement that?
            raise NotImplementedError()
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

    # setup the approach
    params = parameters
    if approach == 'dwa':
        params = {}
        for key in parameters:
            if key in ["sparsity", "bin_sparsity", "alpha", "delta", "lamb", "sbatch", "lr_min", "lr_factor", "lr_patience", "clipgrad"]:
                params[key] = parameters[key]
    appr=appr.Appr(net,nepochs=nepochs,lr=lr,**params)
    print(appr.criterion)
    utils.print_optimizer_config(appr.optimizer)
    print('-'*100)

    # Loop tasks
    acc=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    lss=np.zeros((len(taskcla),len(taskcla)),dtype=np.float32)
    for t,ncla in taskcla:
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

        # Test
        for u in range(t+1):
            xtest=data[u]['test']['x'].cuda()
            ytest=data[u]['test']['y'].cuda()
            test_loss,test_acc=appr.eval(u,xtest,ytest)
            print('>>> Test on task {:2d} - {:15s}: loss={:.3f}, acc={:5.1f}% <<<'.format(u,data[u]['name'],test_loss,100*test_acc))
            acc[t,u]=test_acc
            lss[t,u]=test_loss

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
            with gzip.open(os.path.join(appr.logpath), 'wb') as output:
                pickle.dump(appr.logs, output, pickle.HIGHEST_PROTOCOL)

    ########################################################################################################################


if __name__ == "__main__":
    fire.Fire(main)