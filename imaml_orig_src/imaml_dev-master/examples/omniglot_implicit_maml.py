import numpy as np
import torch
import torch.nn as nn

import os
import sys
sys.path.insert(0, "/u/arushig/imaml_dev-master")


#from measure_accuracy2 import get_acc
import implicit_maml.utils as utils
import random
import time as timer
import pickle
import argparse
import pathlib
import time
import copy
from tqdm import tqdm
from implicit_maml.dataset import OmniglotTask, OmniglotFewShotDataset
from implicit_maml.learner_model import Learner
from implicit_maml.learner_model import make_fc_network, make_conv_network
from implicit_maml.utils import DataLog
from torchvision.transforms import Compose, Resize, ToTensor

from torchmeta.transforms import Categorical, ClassSplitter, Rotation

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader


import matplotlib.pyplot as plt
import wandb



np.random.seed(123)
torch.manual_seed(123)
random.seed(123)
logger = DataLog()



# There are 1623 characters
train_val_permutation = list(range(1623))
random.shuffle(train_val_permutation)

# ===================
# hyperparameters
# ===================
parser = argparse.ArgumentParser(description='Implicit MAML on Omniglot dataset')
#parser.add_argument('--data_dir', type=str, default='/home/aravind/data/omniglot-py/',
                    #help='location of the dataset')
parser.add_argument('--N_way', type=int, default=5, help='number of classes for few shot learning tasks')
parser.add_argument('--K_shot', type=int, default=1, help='number of instances for few shot learning tasks')
parser.add_argument('--inner_lr', type=float, default=1e-2, help='inner loop learning rate')
parser.add_argument('--outer_lr', type=float, default=1e-2, help='outer loop learning rate')
parser.add_argument('--n_steps', type=int, default=16, help='number of steps in inner loop')
parser.add_argument('--meta_steps', type=int, default=1000, help='number of meta steps')
parser.add_argument('--task_mb_size', type=int, default=16)
parser.add_argument('--lam', type=float, default=1.0, help='regularization in inner steps')
parser.add_argument('--cg_steps', type=int, default=5)
parser.add_argument('--cg_damping', type=float, default=1.0)
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--num_tasks', type=int, default=20000)
parser.add_argument('--save_dir', type=str, default='/tmp')
parser.add_argument('--lam_lr', type=float, default=0.0)
parser.add_argument('--lam_min', type=float, default=0.0)
parser.add_argument('--scalar_lam', type=bool, default=True, help='keep regularization as a scalar or diagonal matrix (vector)')
parser.add_argument('--taylor_approx', type=bool, default=False, help='Use Neumann approximation for (I+eps*A)^-1')
parser.add_argument('--inner_alg', type=str, default='gradient', help='gradient or sqp for inner solve')
parser.add_argument('--load_agent', type=str, default=None)
parser.add_argument('--load_tasks', type=str, default=None)
parser.add_argument('--tr_val',  action='store_true' )
parser.add_argument('--nowandb', action='store_true')


args = parser.parse_args()
logger.log_exp_args(args)



data_test_batch_size = 600
os.environ["WANDB_API_KEY"] = 'USE YOUR API KEY'
#wandb.login()
if not args.nowandb:
    wandb.init(name='lambda ' + str(args.lam), project = 'meta_split3', notes='This is a test run', config = vars(args))


print('Generating tasks.......')

#change this to include augmentations
dataset = omniglot("/n/fs/ptml/arushig/data", ways=args.N_way, shots=args.K_shot, test_shots=1, meta_train=True, download=False, transform=Compose([Resize(28), ToTensor()]), target_transform=Categorical(num_classes=args.N_way), class_augmentations=[Rotation([90, 180, 270])])
dataloader = BatchMetaDataLoader(dataset, batch_size=args.task_mb_size)

datasettest = omniglot("/n/fs/ptml/arushig/data", ways=args.N_way, shots=args.K_shot, test_shots=1, meta_test = True, download=False, transform=Compose([Resize(28), ToTensor()]), target_transform=Categorical(num_classes=args.N_way), class_augmentations=[Rotation([90, 180, 270])])
dataloadertest = BatchMetaDataLoader(datasettest, batch_size=data_test_batch_size)
print('Done with data loader')
'''
print("Generating tasks ...... ")
if args.load_tasks is None:
    task_defs = [OmniglotTask(train_val_permutation, root=args.data_dir, num_cls=args.N_way, num_inst=args.K_shot) for _ in tqdm(range(args.num_tasks))]
    dataset = OmniglotFewShotDataset(task_defs=task_defs, GPU=args.use_gpu)
else:
    task_defs = pickle.load(open(args.load_tasks, 'rb'))
    assert args.N_way == task_defs[0].num_cls and args.K_shot == task_defs[0].num_inst and args.num_tasks <= len(task_defs)
    task_defs = task_defs[:args.num_tasks]
    dataset = OmniglotFewShotDataset(task_defs=task_defs, GPU=args.use_gpu)
'''



def get_acc(dloader, lamval, mlearner, falearner, inner_st, trval = 'True'):
    meta_learner = copy.deepcopy(mlearner)
    fastlearner = copy.deepcopy(falearner)
    for learner in [meta_learner, fast_learner]:
        learner.inner_alg = args.inner_alg
        learner.set_inner_lr(args.inner_lr)
    lam = torch.tensor(lamval) #torch.tensor(args.lam)
    lam = lam.to(device)

    losses = np.zeros((data_test_batch_size, 5))
    accuracy = np.zeros((data_test_batch_size, 2))
    meta_params = meta_learner.get_params().clone()

    test_minibatch = next(iter(dloader))
    test_train_inp, test_train_targ = test_minibatch['train']
    test_test_inp, test_test_targ = test_minibatch['test']
    for i, (x_inp, y_inp, v_inp, vy_inp) in enumerate( zip( test_train_inp, test_train_targ, test_test_inp, test_test_targ )):
        #for i in tqdm(range(args.num_tasks)):
        fast_learner.set_params(meta_params)
        #task = dataset.__getitem__(i)
        task = {'x_train': x_inp.to(device), 'y_train': y_inp.to(device), 'x_val': v_inp.to(device), 'y_val' : vy_inp.to(device)}
        if trval:
            vl_before = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
        else:
            bothx = torch.cat(( task['x_train'], task['x_val']), 0)
            bothy = torch.cat((  task['y_train'], task['y_val']), 0)
            vl_before = fast_learner.get_loss(bothx, bothy, return_numpy=True)
        tl = fast_learner.learn_task(task, num_steps=inner_st, add_regularization=False, tr_val = trval)
        fast_learner.inner_opt.zero_grad()
        regu_loss = fast_learner.regularization_loss(meta_params, args.lam)
        regu_loss.backward()
        fast_learner.inner_opt.step()
        if trval:
            train_loss_after = fast_learner.get_loss(task['x_train'], task['y_train'], return_numpy=True)
        else:
            bothx = torch.cat( ( task['x_train'], task['x_val']), 0)
            bothy = torch.cat( ( task['y_train'], task['y_val']), 0)
            #note that this captures loss after learning task and regularization, not before regularization
            train_loss_after = fast_learner.get_loss(bothx, bothy, return_numpy=True)
        if trval:
            vl_after = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
        else:
             bothx = torch.cat( ( task['x_train'], task['x_val']), 0)
             bothy = torch.cat( ( task['y_train'], task['y_val']), 0)
             vl_after = fast_learner.get_loss(bothx, bothy, return_numpy=True)
        if trval:
            tacc = utils.measure_accuracy(task, fast_learner, train=True)
            vacc = utils.measure_accuracy(task, fast_learner, train=False)
        else:
            tacc = utils.measure_accuracy(task, fast_learner, train=True, useboth=True)
            vacc = utils.measure_accuracy(task, fast_learner, train=False, useboth=True)
            assert(tacc==vacc)
        losses[i] = np.array([tl[0], vl_before, tl[-1], vl_after, train_loss_after])
        accuracy[i][0] = tacc; accuracy[i][1] = vacc

    #print("Mean accuracy: on train set and val set on specified tasks (meta-train or meta-test)")
    #print(np.mean(accuracy, axis=0))
    #print("95% confidence intervals for the mean")
    #print(1.96*np.std(accuracy, axis=0)/np.sqrt(args.num_tasks))  # note that accuracy is already in percentage
    #print('Loss after ',   np.mean(losses, axis = 0)[-1]  )
    return (np.mean(accuracy, axis=0), 1.96*np.std(accuracy, axis=0)/np.sqrt(args.num_tasks), np.mean(losses, axis = 0)[-1]  , np.mean(losses, axis = 0)[3] )









if args.load_agent is None:
    learner_net = make_conv_network(in_channels=1, out_dim=args.N_way)
    fast_net = make_conv_network(in_channels=1, out_dim=args.N_way)
    meta_learner = Learner(model=learner_net, loss_function=torch.nn.CrossEntropyLoss(), inner_alg=args.inner_alg,
                           inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu)
    fast_learner = Learner(model=fast_net, loss_function=torch.nn.CrossEntropyLoss(), inner_alg=args.inner_alg,
                           inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu)
    fast_learner2 = Learner(model=fast_net, loss_function=torch.nn.CrossEntropyLoss(), inner_alg=args.inner_alg,
                           inner_lr=args.inner_lr, outer_lr=args.outer_lr, GPU=args.use_gpu)
else:
    meta_learner = pickle.load(open(args.load_agent, 'rb'))
    meta_learner.set_params(meta_learner.get_params())
    fast_learner = pickle.load(open(args.load_agent, 'rb'))
    fast_learner.set_params(fast_learner.get_params())
    for learner in [meta_learner, fast_learner]:
        learner.inner_alg = args.inner_alg
        learner.inner_lr = args.inner_lr
        learner.outer_lr = args.outer_lr
    
init_params = meta_learner.get_params()
device = 'cuda' if args.use_gpu is True else 'cpu'
lam = torch.tensor(args.lam) if args.scalar_lam is True else torch.ones(init_params.shape[0])*args.lam
lam = lam.to(device)

pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# ===================
# Train
# ===================
print("Training model ......")
print(args.tr_val)
losses = np.zeros((args.meta_steps, 4))
accuracy = np.zeros((args.meta_steps, 2))
#for outstep in tqdm(range(args.meta_steps)):
with tqdm(dataloader, total=args.meta_steps) as pbar:
    for outstep, task_mb in enumerate(pbar):
        #print('outstep', outstep)
        #task_mb = np.random.choice(args.num_tasks, size=args.task_mb_size)
        w_k = meta_learner.get_params()
        meta_grad = 0.0
        lam_grad = 0.0
        #print('about to process batch', time.time())
        sys.stdout.flush()
        #check how handle cuda
        train_inputs, train_targets = task_mb['train']
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)

        val_inputs, val_targets = task_mb['test']
        val_inputs = val_inputs.to(device)
        val_targets = val_targets.to(device)
        

        for task_idx, (train_inp, train_targ, val_inp, val_targ) in enumerate(zip(train_inputs, train_targets, val_inputs, val_targets)):#in task_mb:
       


            task={ 'x_train': train_inp,
                   'x_val': val_inp,
                   'y_train': train_targ,
                   'y_val' : val_targ
            } 
            fast_learner.set_params(w_k.clone()) # sync weights
            #print(idx)
            #task = dataset.__getitem__(idx) # get task
            if args.tr_val:
                vl_before = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
            else:
                bothx = torch.cat(( task['x_train'], task['x_val']), 0)
                bothy = torch.cat((  task['y_train'], task['y_val']), 0)
                vl_before = fast_learner.get_loss(bothx, bothy, return_numpy=True)



            #have to modify learn_task
            #metatorch feed task dictionary
            tl = fast_learner.learn_task(task, num_steps=args.n_steps, tr_val = args.tr_val)
            # pull back for regularization
            fast_learner.inner_opt.zero_grad()

            #regularization
            regu_loss = fast_learner.regularization_loss(w_k, lam)
            regu_loss.backward()
            fast_learner.inner_opt.step()
            if args.tr_val:
                vl_after = fast_learner.get_loss(task['x_val'], task['y_val'], return_numpy=True)
            else:
                bothx = torch.cat( ( task['x_train'], task['x_val']), 0)
                bothy = torch.cat( ( task['y_train'], task['y_val']), 0)
                vl_after = fast_learner.get_loss(bothx, bothy, return_numpy=True)
       
            #If using tr tr we will set these to the same value evaluated on all the data

            if args.tr_val:
                tacc = utils.measure_accuracy(task, fast_learner, train=True)
                vacc = utils.measure_accuracy(task, fast_learner, train=False)
            else:
                tacc = utils.measure_accuracy(task, fast_learner, train=True, useboth=True)
                vacc = utils.measure_accuracy(task, fast_learner, train=False, useboth=True)
                assert(tacc==vacc)

            if args.tr_val:
                valid_loss = fast_learner.get_loss(task['x_val'], task['y_val'])
            else:
                bothx = torch.cat( ( task['x_train'], task['x_val']), 0)
                bothy = torch.cat( ( task['y_train'], task['y_val']), 0)
                valid_loss = fast_learner.get_loss(bothx, bothy)


            valid_grad = torch.autograd.grad(valid_loss, fast_learner.model.parameters())
            flat_grad = torch.cat([g.contiguous().view(-1) for g in valid_grad])
        
            if args.cg_steps <= 1:
                task_outer_grad = flat_grad
            else:
                task_matrix_evaluator = fast_learner.matrix_evaluator(task, lam, args.cg_damping, tr_val=args.tr_val)
                task_outer_grad = utils.cg_solve(task_matrix_evaluator, flat_grad, args.cg_steps, x_init=None)
        
            meta_grad += (task_outer_grad/args.task_mb_size)
            losses[outstep] += (np.array([tl[0], vl_before, tl[-1], vl_after])/args.task_mb_size)
            accuracy[outstep] += np.array([tacc, vacc]) / args.task_mb_size
              
            if args.lam_lr <= 0.0:
                task_lam_grad = 0.0
            else:
                print("Warning: lambda learning is not tested for this version of code")
                train_loss = fast_learner.get_loss(task['x_train'], task['y_train'])
                train_grad = torch.autograd.grad(train_loss, fast_learner.model.parameters())
                train_grad = torch.cat([g.contiguous().view(-1) for g in train_grad])
                inner_prod = train_grad.dot(task_outer_grad)
                task_lam_grad = inner_prod / (lam**2 + 0.1)

            lam_grad += (task_lam_grad / args.task_mb_size)
        #print('end of batch process', time.time())
        sys.stdout.flush()
        meta_learner.outer_step_with_grad(meta_grad, flat_grad=True)
        lam_delta = - args.lam_lr * lam_grad
        lam = torch.clamp(lam + lam_delta, args.lam_min, 5000.0) # clips each element individually if vector
        logger.log_kv('train_pre', losses[outstep,0])
        logger.log_kv('test_pre', losses[outstep,1])
        logger.log_kv('train_post', losses[outstep,2])
        logger.log_kv('test_post', losses[outstep,3])
        logger.log_kv('train_acc', accuracy[outstep, 0])
        logger.log_kv('val_acc', accuracy[outstep, 1])
        if not args.nowandb:
            wandb.log({ 
                'train_pre' : losses[outstep, 0],
                'vl_pre' : losses[outstep, 1],
                'train_post' : losses[outstep, 2],
                'vl_post' : losses[outstep, 3],
                'train_acc' : accuracy[outstep, 0],
                'val_acc' : accuracy[outstep, 1]
                })

        if (outstep % 50 == 0 and outstep > 0) or outstep == args.meta_steps-1:
            smoothed_losses = utils.smooth_vector(losses[:outstep], window_size=10)
            plt.figure(figsize=(10,6))
            plt.plot(smoothed_losses)
            plt.ylim([0, 2.0])
            plt.xlim([0, args.meta_steps])
            plt.grid(True)
            plt.legend(['Train pre', 'Test pre', 'Train post', 'Test post'], loc=1)
            plt.savefig(args.save_dir+'/learn_curve.png', dpi=100)
            plt.clf()
            plt.close('all')
        
            smoothed_acc = utils.smooth_vector(accuracy[:outstep], window_size=25)
            plt.figure(figsize=(10,6))
            plt.plot(smoothed_acc)
            plt.ylim([50.0, 100.0])
            plt.xlim([0, args.meta_steps])
            plt.grid(True)
            plt.legend(['Train post', 'Test post'], loc=4)
            plt.savefig(args.save_dir+'/accuracy.png', dpi=100)
            plt.clf()
            plt.close('all')

            pickle.dump(meta_learner, open(args.save_dir+'/agent.pickle', 'wb'))
            logger.save_log()
  



        if (outstep % 500 == 0 and outstep > 0):
            '''
            print('test accuracy')
            #obviously this is bad, but we want to show that even if we do this tr-tr doesn't work
            losses_test = np.zeros((args.num_tasks, 4))
            accuracy_test = np.zeros((args.num_tasks, 2))
            meta_params = meta_learner.get_params().clone()#.to(device)
            test_minibatch = next(iter(dataloadertest))
            #print(test_minibatch)
            #print(np.shape(test_minibatch))

            test_train_inp, test_train_targ = test_minibatch['train']
            #test_train_inp.to(device)
            #test_train_targ.to(device)

            test_test_inp, test_test_targ = test_minibatch['test']
            #test_test_inp.to(device)
            #test_test_targ.to(device)
            #fast_learner2 = fast_learner.clone) 
            for i, (x_inp, y_inp, v_inp, vy_inp) in enumerate( zip( test_train_inp, test_train_targ, test_test_inp, test_test_targ )):#i in tqdm(range(args.num_tasks)):
                fast_learner2.set_params(meta_params)
                task = {'x_train': x_inp.to(device), 'y_train': y_inp.to(device), 'x_val': v_inp.to(device), 'y_val' : vy_inp.to(device)} 
                
               #dataset.__getitem__(i)
                vl_before_test = fast_learner2.get_loss(task['x_val'], task['y_val'], return_numpy=True)
                tl = fast_learner2.learn_task(task, num_steps=args.n_steps, add_regularization=False)
                fast_learner2.inner_opt.zero_grad()
                regu_loss = fast_learner2.regularization_loss(meta_params, args.lam)
                regu_loss.backward()
                fast_learner2.inner_opt.step()
                vl_after_test = fast_learner2.get_loss(task['x_val'], task['y_val'], return_numpy=True)
                tacc2 = utils.measure_accuracy(task, fast_learner2, train=True)
                vacc2 = utils.measure_accuracy(task, fast_learner2, train=False)
                losses_test[i] = np.array([tl[0], vl_before_test, tl[-1], vl_after_test])
                accuracy_test[i][0] = tacc2; accuracy_test[i][1] = vacc2

                #print("Mean accuracy: on train set and val set on specified tasks (meta-train or meta-test)")
                #print(np.mean(accuracy, axis=0))
                #print("95% confidence intervals for the mean")
                #print(1.96*np.std(accuracy, axis=0)/np.sqrt(args.num_tasks))  # note that accuracy is already in percentage
            print('meta train', np.mean(accuracy_test, axis =0)[0])
            '''
            accy , _, lotrainrv, lotvalrv = get_acc(dataloadertest, args.lam , meta_learner, fast_learner , args.n_steps, trval = True)
            accytt , _, lotrainrvtt, lotvalrvtt = get_acc(dataloadertest, args.lam , meta_learner, fast_learner , args.n_steps, trval = False)
            if not args.nowandb:
                wandb.log({'meta_test_acc_trval' : accy[1] , 'meta_test_train_acc_trval': accy[0], 'meta_test_loss_trval' : lotrainrv, 'meta_test_train_loss_trval' : lotvalrv, 'meta_test_trtracc' : accytt[0], 'meta_test_trtrloss' : lotrainrvtt}  )


        if (outstep % 1000 == 0 and outstep > 0):
            checkpoint_file = args.save_dir + '/checkpoint_' + str(outstep) + '.pickle'
            pickle.dump(meta_learner, open(checkpoint_file, 'wb'))
            #have wandb save the checkpoint model
            #wandb.save('checkpoint_file')

        if outstep == args.meta_steps-1:
            checkpoint_file = args.save_dir + '/final_model.pickle'
            pickle.dump(meta_learner, open(checkpoint_file, 'wb'))
