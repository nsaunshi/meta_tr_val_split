import os
import sys
from time import time
from torch import optim
import yaml


sys.path.append('/n/fs/ptml/arushig/miniimagenet/meta_split/src/')
from learners import RepLearner, iMAMLLearner, MetaProxLearner
from data_loaders import sine_loaders, sine_kshotNway, simul_loaders, simul_kshotNway
from data_loaders import omniglot_loaders, omniglot_kshotNway, miniimagenet_loaders, miniimagenet_kshotNway
from models import SineModel, LinearRepModel, OmniglotModel, MiniImagenetModel, MetaLearningModel, OmniglotModelFC, MiniImagenetModelWidth
from utils import DotDict, get_device, mkdir, get_latest_checkpoint, set_seed
import time
import numpy as np 
from numpy import unravel_index
import torch
from torch import nn
import torch.nn.functional as F
import torchmeta




config = yaml.load(open(sys.argv[1], 'r'), Loader=yaml.FullLoader)


print('model dir is ', sys.argv[2])
checkpoint_dir = sys.argv[2] 
args = DotDict(config)

sys.stdout.flush()
 ## Set appropriate random seeds
set_seed(args)


## Do we need model to be end-to-end?
if args.learner.lower() == 'replearn':
    e2e = False
else:
    e2e = True


if args.dataset.lower() == 'omni' or args.dataset.lower() == 'omniglot':
    if args.usefc:
        print('nww', args.netwidth)
        loaders = omniglot_loaders(args)
        model = OmniglotModelFC(args.N_way, e2e=e2e, classifier=args.classifier, net_width = args.netwidth)
        regression = False
    else:
        loaders = omniglot_loaders(args)
        model = OmniglotModel(args.N_way, e2e=e2e, classifier=args.classifier)
        regression = False


if args.dataset.lower() == 'mini' or args.dataset.lower() == 'miniimagenet':
    if args.usefc:
        loaders = miniimagenet_loaders(args)
        #same baseline net we used for omni
        model = MiniModelFC(args.N_way, e2e=e2e, classifier=args.classifier,  net_width = args.netwidth)
        regression = False
    elif args.useconv:
        loaders = miniimagenet_loaders(args)
        model = MiniImagenetModelWidth(args.N_way, e2e=e2e, classifier=args.classifier,  net_width = args.netwidth)
        regression = False
    else:
        loaders = miniimagenet_loaders(args)
        model = MiniImagenetModel(args.N_way, e2e=e2e, classifier=args.classifier)
        regression = False




# Set optimizer and learner
if args.learner.lower() == 'replearn':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    learner = RepLearner(args, model, optimizer, regression=regression)


learner.to('cuda')

learner.load_checkpoint(checkpoint_dir)

# losses at 0 and accs at 1
lambda_grid = [0, .1 , .3, 1. , 2., 3., 10., 100.]
inner_step_grid  = [50 ,100, 200]
lamaccs = np.zeros((len(lambda_grid), len(inner_step_grid)))
for instep in range(len(inner_step_grid)):
    for lval in range(len(lambda_grid)):
        #print(time.time())
        print(str(instep), str(lval))
        ans = learner.eval_model(learner.model, loaders[1], variant ='tr-te', stop_eval = 20, regression=False, mean = True, firstorder=True, transductive = True, reg = lambda_grid[lval], inner_steps = inner_step_grid[instep], custom = True, device = 'cuda')
        lamaccs[lval][instep]= ans[1]
        #print(time.time())

print(lamaccs)

best_indcs = unravel_index(lamaccs.argmax(), lamaccs.shape)
print('best lambda value on val set was ', str(lambda_grid[best_indcs[0]]), 'and step ', str(inner_step_grid[best_indcs[1]]) )#str(lambda_grid[np.argmax(lamaccs)]))
best_lam = lambda_grid[best_indcs[0]]
best_in = inner_step_grid[best_indcs[1]]

#ok, let's just reload the learner to reset the stats
learner.load_checkpoint(checkpoint_dir)

ll, accl = learner.eval_model(learner.model, loaders[2], variant ='tr-te', stop_eval = 40, regression=False, firstorder=True, transductive = True, reg = best_lam, inner_steps = best_in, mean=False, custom=True, device = 'cuda')



if  args.usefc:
    f = open('../test_time_accsfc/' + sys.argv[1].split('/')[-1], 'w')
elif args.dataset.lower() == 'mini' or args.dataset.lower() == 'miniimagenet':
    f = open('../test_time_accsmini/' + sys.argv[2].split('/')[-2], 'w')
else:
    f = open('../test_time_accs/' + sys.argv[1].split('/')[-1], 'w')

f.write('{0:.6f}'.format(np.mean(accl)) + '\t' + '{0:.6f}'.format( 1.96*np.std(accl)/np.sqrt(len(accl)) ) + '\n')
f.close()


ll, accl = learner.eval_model(learner.model, loaders[2], variant ='tr-tr', stop_eval = 20, regression=False, reg = best_lam, inner_steps = best_in, mean = False, custom=True)

