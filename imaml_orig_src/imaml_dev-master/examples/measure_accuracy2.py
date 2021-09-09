import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.insert(0, "/u/arushig/imaml_dev-master")
sys.path.insert(0, "/home/user/Desktop/meta-learning/data/imaml_dev-master")

import implicit_maml.utils as utils
import random
import time as timer
import pickle
import argparse
import os

from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader


from tqdm import tqdm
from implicit_maml.dataset import OmniglotTask, OmniglotFewShotDataset
from implicit_maml.learner_model import Learner
from implicit_maml.learner_model import make_fc_network, make_conv_network
from implicit_maml.utils import DataLog
import copy
from numpy import unravel_index
from torchvision.transforms import Compose, Resize, ToTensor

from torchmeta.transforms import Categorical, ClassSplitter, Rotation


SEED = 12345

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# There are 1623 characters
train_val_permutation = list(range(1623))
random.shuffle(train_val_permutation)

# ===================
# hyperparameters
# ===================
parser = argparse.ArgumentParser(description='Parameters for test performance')
parser.add_argument('--task', type=str, default='Omniglot')
parser.add_argument('--data_dir', type=str, default='/home/aravind/data/omniglot-py/',
                    help='location of the dataset')
parser.add_argument('--train_set', action='store_true')  # defaults to False if nothing is provided to the argument, if --train_set is passed on, then would become true
parser.add_argument('--N_way', type=int, required=True)
parser.add_argument('--K_shot', type=int, required=True)
parser.add_argument('--num_tasks', type=int, default=600)

parser.add_argument('--load_agent', type=str, required=True)
parser.add_argument('--n_steps', type=int, default=5)
parser.add_argument('--lam', type=float, default=0.0)
parser.add_argument('--inner_lr', type=float, default=1e-2)
parser.add_argument('--inner_alg', type=str, default='gradient')
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--tr_val', type=bool, default=True)
args = parser.parse_args()
'''
assert args.task == 'Omniglot' or args.task == 'MiniImageNet'
print(args)

print("Generating tasks ...... ")
print("Meta-training set? : ", str(args.train_set))
task_defs = [OmniglotTask(train_val_permutation, root=args.data_dir, num_cls=args.N_way, num_inst=args.K_shot, train=args.train_set) for _ in tqdm(range(args.num_tasks))]
dataset = OmniglotFewShotDataset(task_defs=task_defs, GPU=args.use_gpu)
'''

#get_acc(dataloadertest, best_lam, meta_learner, fast_learner )
#trval will bbe true if we're getting trval loss, else we're gettign the train trian loss
def get_acc(dloader, lamval, mlearner, falearner, inner_st, trval = 'True'):
    meta_learner = copy.deepcopy(mlearner)
    fastlearner = copy.deepcopy(falearner)
    for learner in [meta_learner, fast_learner]:
        learner.inner_alg = args.inner_alg
        learner.set_inner_lr(args.inner_lr)
    lam = torch.tensor(lamval) #torch.tensor(args.lam)
    lam = lam.to(device)

    losses = np.zeros((args.num_tasks, 5))
    accuracy = np.zeros((args.num_tasks, 2))
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
        tl = fast_learner.learn_task(task, num_steps=inner_st, add_regularization=False, tr_val=trval)
        fast_learner.inner_opt.zero_grad()
        regu_loss = fast_learner.regularization_loss(meta_params, lamval)
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

    print("Mean accuracy: on train set and val set on specified tasks (meta-train or meta-test)")
    print(np.mean(accuracy, axis=0))
    print("95% confidence intervals for the mean")
    print(1.96*np.std(accuracy, axis=0)/np.sqrt(args.num_tasks))  # note that accuracy is already in percentage
    print('Loss after ',   np.mean(losses, axis = 0)[-1]  )
    return (np.mean(accuracy, axis=0), 1.96*np.std(accuracy, axis=0)/np.sqrt(args.num_tasks), np.mean(losses, axis = 0)[-1] , np.mean(losses, axis = 0)[3] )
#        f = open("lam0.5.txt", "a")

 #       f.write(iterno +  ' ' +  str(np.mean(accuracy, axis =0)[1]) + '\n')
 #       f.close()
                                   

datasettest = omniglot("/n/fs/ptml/arushig/data", ways=args.N_way, shots=args.K_shot, test_shots=1, meta_test = True, download=False, transform=Compose([Resize(28), ToTensor()]), target_transform=Categorical(num_classes=args.N_way), class_augmentations=[Rotation([90, 180, 270])])
dataloadertest = BatchMetaDataLoader(datasettest, batch_size=600)


#let's just change this back to evaluate on args.load_agent
#for fi in os.listdir('/n/fs/ptml/arushig/5_way_1_shot_lam0.5'):
#if 'checkpoint' in fi:
#iterno = fi.split('/')[-1].split('_')[1].split('.')[0]
print("Loading agent")
meta_learner = pickle.load(open(args.load_agent, 'rb'))
meta_learner.set_params(meta_learner.get_params())
fast_learner = pickle.load(open(args.load_agent, 'rb'))
fast_learner.set_params(fast_learner.get_params())
for learner in [meta_learner, fast_learner]:
    learner.inner_alg = args.inner_alg
    learner.set_inner_lr(args.inner_lr)

init_params = meta_learner.get_params()
device = 'cuda' if args.use_gpu is True else 'cpu'


#Ok here's where we call upon the val set 

datasetval = omniglot("/n/fs/ptml/arushig/data", ways=args.N_way, shots=args.K_shot, test_shots=1, meta_val = True, download=False, transform=Compose([Resize(28), ToTensor()]), target_transform=Categorical(num_classes=args.N_way), class_augmentations=[Rotation([90, 180, 270])])
dataloaderval = BatchMetaDataLoader(datasetval, batch_size=600)


lambda_grid = [0., .1,.3,  1., 2., 3., 10., 100]#, .3, 3., 5.]
inner_step_grid  = [8 ,16, 32, 64]
lamaccs = np.zeros((len(lambda_grid), len(inner_step_grid)))
for instep in range(len(inner_step_grid)):
    for lval in range(len(lambda_grid)):
        print('trying lambda value', lambda_grid[lval], 'and inner step value ', inner_step_grid[instep])
        accl, _ , _, _ = get_acc(dataloaderval, lambda_grid[lval], meta_learner, fast_learner, inner_step_grid[instep])
        lamaccs[lval][instep] = accl[1]

print(lamaccs)

best_indcs = unravel_index(lamaccs.argmax(), lamaccs.shape)
print('best lambda value on val set was ', str(lambda_grid[best_indcs[0]]), 'and step ', str(inner_step_grid[best_indcs[1]]) )#str(lambda_grid[np.argmax(lamaccs)]))
best_lam = lambda_grid[best_indcs[0]]
best_in = inner_step_grid[best_indcs[1]]

print('and the meta-test trval acc is ')
accm, accstd, _, _ = get_acc(dataloadertest, best_lam, meta_learner, fast_learner , best_in)
print('accm', accm)
print('acc std', accstd)

fistr = ''
if 'relaunch' in args.load_agent.split('/')[-2]:
    fistr = args.load_agent.split('/')[-3]
else:
    fistr = args.load_agent.split('/')[-2]
f = open('../test_accs/'+ fistr +'.txt', 'w')
f.write('{0:.4f}'.format(accm[1]) + '\t' + '{0:.4f}'.format(accstd[1]) + '\n')
f.close()

