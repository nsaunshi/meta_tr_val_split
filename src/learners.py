import numpy as np
import os
from time import time
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
try:
    mp.set_start_method("spawn")
except:
    pass

from inner_loops import inner_loop_iMAML, inner_loop_RepLearn, inner_loop_MetaProx
from utils import apply_grad, grad_means, get_device, copy_model, params_to_vec, copy_params, vec_to_params




class Learner:
    '''
    Base class for different learners
    '''
    def __init__(self, args, model, optimizer):
        self.args = args
        self.model = model
        self.optimizer = optimizer

    def checkpoint(self, iteration, final=False):
        args = self.args
        model = self.model
        optimizer = self.optimizer

        checkpoint_name = 'checkpoint-{0}' + final*'-final' + '.ckpt'
        if self.args.debug:
            print('Saving checkpoint to {0}'.format(checkpoint_name.format(iteration)))
        torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(args.save_dir,checkpoint_name.format(iteration)))

    def load_checkpoint(self, checkpoint):
        if self.args.debug:
            print('Loading from checkpoint {0}'.format(checkpoint))
        device = get_device(self.args.device)
        checkpoint_data = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(checkpoint_data['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])

    def log(self, log_data, iteration=''):
        '''
        Log the evaluation data for particular iteration on weights and biases if needed and print if needed.
        '''
        args = self.args
        # Log values on weights and biases
        if args.wandb:
            import wandb
            for data in log_data:
                wandb.log(data, step=iteration)

        # Print values
        if args.debug:
            print_string = '{0}: '.format(iteration)
            for data in log_data:
                print_string += '(' + ', '.join(['{0}: {1:.4f}'.format(key, value) for key, value in data.items()]) + '), '
            print(print_string)
        return

    def to(self, device):
        self.model.to(device)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def step(self, batch):
        raise NotImplementedError

    def evaluate(self, iteration):
        raise NotImplementedError

    @staticmethod
    def batch_embed(model, data):
        X, Y = data
        device = next(model.parameters()).device
        if X.dtype == torch.double:
            X = X.float()
        if Y.dtype == torch.double:
            Y = Y.float()
        return X.to(device), Y.to(device)




class RepLearner(Learner):
    '''
    Representation learning learner (for both regression and classification) that handles both the tr-tr and tr-te variants
    TODO: Implement transductive setting
    '''
    def __init__(self, args, model, optimizer, regression=False):
        super().__init__(args, model, optimizer)
        self.regression = regression


    def step(self, batch):
        args = self.args
        optimizer = self.optimizer
        model = self.model

        optimizer.zero_grad()
        tr_Xs, tr_Ys = batch['train']
        te_Xs, te_Ys = batch['test']

        # Embed tr and te data into representations using model
        tr_Xs, tr_Ys = self.batch_embed(model, (tr_Xs, tr_Ys))
        te_Xs, te_Ys = self.batch_embed(model, (te_Xs, te_Ys))

        kwargs = {'reg': args.reg, 'custom': args.custom, 'inner_lr': args.inner_lr,
                  'inner_steps': args.train_inner_steps, 'device': args.device}
        loss = inner_loop_RepLearn(
            model, (tr_Xs, tr_Ys), (te_Xs, te_Ys), variant=args.variant, firstorder=args.firstorder,
            grad=True, acc=False, regression=self.regression, **kwargs
        )

        loss.backward()
        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()


    def evaluate(self, loaders_info, mean=True, transductive=True):
        '''
        Takes list of (string, bool, loader) tuples and outputs a dictionary of output values
        TODO: add mean and std
        TODO: implement transductive
        '''
        model = self.model
        args = self.args
        inner_loop_kwargs = {'firstorder': args.firstorder, 'reg': args.reg, 'custom': args.custom, 'inner_lr': args.inner_lr,
                             'inner_steps': args.eval_inner_steps, 'device': args.device, 'transductive': transductive}
        eval_data = []

        for loader_info in loaders_info:
            prefix, tr_te_only, loader = loader_info
            data = {}
            
            loss_tr_te, acc_tr_te = self.eval_model(
                model, loader, stop_eval=args.stop_eval,
                variant='tr-te', regression=self.regression, mean=mean, **inner_loop_kwargs
            )
            data[prefix + '_loss_tr_te'] = loss_tr_te
            if not self.regression:
                data[prefix + '_acc_tr_te'] = 100*acc_tr_te
                
            if not tr_te_only:
                loss_tr_tr, acc_tr_tr = self.eval_model(
                    model, loader, stop_eval=args.stop_eval,
                    variant='tr-tr', regression=self.regression, mean=mean, **inner_loop_kwargs
                )
                data[prefix + '_loss_tr_tr'] = loss_tr_tr
                if not self.regression:
                    data[prefix + '_acc_tr_tr'] = 100*acc_tr_tr

            eval_data.append(data)
        return eval_data
    
    
    @classmethod
    def eval_model(cls, model, loader, stop_eval=20, variant='tr-tr', regression=False, mean=True, **inner_loop_kwargs):
        '''
        Evaluate the model. Defined as a separate classmethod to be able to evaluate a model independently with different settings for stop_eval, mean, inner_loop_kwargs
        inner_loop_kwargs: Should cover {'firstorder': args.firstorder, 'reg': args.reg, 'custom': args.custom, 'inner_lr': args.inner_lr,
                             'inner_steps': args.eval_inner_steps, 'device': args.device, 'transductive': transductive}
        Example usage:
        eval_model(cls, model, loader, stop_eval=20, variant='tr-te', regression=False, mean=False, firstorder=True, reg=1.0, custom=True, inner_lr=0.05, inner_steps=100, device='cuda:0', transductive=True)
        eval_model(cls, model, loader, stop_eval=20, variant='tr-te', regression=True, mean=False, firstorder=True, reg=1.0, custom=True, inner_lr=0.05, inner_steps=100, device='cuda:0', transductive=True)
        '''

        loss = 0.
        acc = 0.
        loss_list = []
        acc_list = []
        total = 0.
        model.train()
        with torch.no_grad():
            for i, batch in enumerate(loader):
                # Stop eval if needed
                if not stop_eval is None and i >= stop_eval:
                    break

                tr_Xs, tr_Ys = batch['train']
                te_Xs, te_Ys = batch['test']

                # Embed tr and te data into representations using model
                tr_Xs, tr_Ys = cls.batch_embed(model, (tr_Xs, tr_Ys))
                te_Xs, te_Ys = cls.batch_embed(model, (te_Xs, te_Ys))
                
                output = inner_loop_RepLearn(
                    model, (tr_Xs, tr_Ys), (te_Xs, te_Ys), variant=variant,
                    grad=False, acc=not regression, mean=mean, regression=regression, **inner_loop_kwargs
                )

                # Assign an arbitrary value to acc if this is a regression task
                if regression:
                    l = output
                    a = torch.tensor([-1.])
                else:
                    l, a = output

                if mean:
                    loss += l.item()
                    acc += a.item()
                    total += 1
                else:
                    loss_list.append(l.detach().cpu().numpy())
                    acc_list.append(a.detach().cpu().numpy())

        if mean:
            return loss / total, acc / total
        else:
            return np.hstack(loss_list), np.hstack(acc_list)



class iMAMLLearner(Learner):
    '''
    Implement iMAML, both in tr-tr and tr-te settings
    step:
        1. For each task (TODO: batch tasks), copy OG model (model) into a side model (task_model)
        2. Train task_model for args.inner_step number of steps on the "train data"
        3. For task_model, set up Hessian-vector function (Hv) and thus (Av), where A=I+H/reg
        4. Get the gradient g for task_model on "test data" and learn min_w |(I+H/reg)w - g|^2 using conjugate gradient method
        5. Set gradient for model to w using pytorch magic
        6. optimizer.step()
        7. Pray to the CUDA gods
    '''
    def __init__(self, args, model, optimizer, regression=False):
        super().__init__(args, model, optimizer)
        self.regression = regression


    def step(self, batch):
#         start = time()
        args = self.args
        optimizer = self.optimizer
        model = self.model

        tr_Xs, tr_Ys = batch['train']
        te_Xs, te_Ys = batch['test']
        tr_Xs, tr_Ys = self.batch_embed(model, (tr_Xs, tr_Ys))
        te_Xs, te_Ys = self.batch_embed(model, (te_Xs, te_Ys))

        # Iterate over all tasks
        optimizer.zero_grad()
        model.train()
        grad_mean = 0.

        model_params = params_to_vec(model.parameters()).detach().clone()
#         inner_start = time()
        if not args.parallel:
            for i in range(len(tr_Xs)):
                task_model = copy_model(model, grad=False)
                model_grad = inner_loop_iMAML(
                    task_model, (tr_Xs[i], tr_Ys[i]), (te_Xs[i], te_Ys[i]),
                    variant=args.variant, grad=True, acc=False, reg=args.reg, reg_damp=args.reg_damp,
                    regression=self.regression, reg_params=model_params, inner_lr=args.inner_lr,
                    inner_steps=args.inner_steps, cg_steps=args.cg_steps
                )
                grad_mean += model_grad / len(tr_Xs)

        grad_mean = vec_to_params(model, grad_mean)

        optimizer.zero_grad()
        apply_grad(model, grad_mean)
        clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
#         print('Step time: {0:.4f}'.format(time() - start))


    def evaluate(self, loaders_info, mean=True, transductive=True):
        '''
        Takes list of (string, bool, loader) tuples and outputs a dictionary of output values
        TODO: add mean and std
        '''
#         start = time()
        model = self.model
        args = self.args

        inner_loop_kwargs = {
            'reg': args.reg, 'reg_damp': args.reg_damp, 'inner_lr': args.inner_lr,
            'inner_steps': args.inner_steps, 'cg_steps': args.cg_steps, 'transductive': transductive
        }
        eval_data = []

        for loader_info in loaders_info:
            prefix, tr_te_only, loader = loader_info
            data = {}
            
            loss_tr_te, acc_tr_te = self.eval_model(
                model, loader, stop_eval=args.stop_eval,
                variant='tr-te', regression=self.regression, mean=mean, **inner_loop_kwargs
            )
            data[prefix + '_loss_tr_te'] = loss_tr_te
            if not self.regression:
                data[prefix + '_acc_tr_te'] = 100*acc_tr_te

            if not tr_te_only:
                loss_tr_tr, acc_tr_tr = self.eval_model(
                    model, loader, stop_eval=args.stop_eval,
                    variant='tr-tr', regression=self.regression, mean=mean, **inner_loop_kwargs
                )
                data[prefix + '_loss_tr_tr'] = loss_tr_tr
                if not self.regression:
                    data[prefix + '_acc_tr_tr'] = 100*acc_tr_tr

            eval_data.append(data)
#         print('Eval time: {0:.4f}'.format(time() - start))
        return eval_data
    
    
    @classmethod
    def eval_model(cls, model, loader, stop_eval=20, variant='tr-tr', regression=False, mean=True, **inner_loop_kwargs):
        '''
        Evaluate the model. Defined as a separate classmethod to be able to evaluate a model independently with different settings for stop_eval, mean, inner_loop_kwargs
        inner_loop_kwargs: Need to cover the following arguments (reg=1.0, inner_lr=0.01, inner_steps=20, cg_steps=5)
        '''

        loss = 0.
        acc = 0.
        loss_list = []
        acc_list = []
        total = 0.
        model_params = params_to_vec(model.parameters()).detach().clone()
        
        with torch.no_grad():
            for i, batch in enumerate(loader):
                # Stop eval if needed
                if not stop_eval is None and i >= stop_eval:
                    break

                tr_Xs, tr_Ys = batch['train']
                te_Xs, te_Ys = batch['test']
                tr_Xs, tr_Ys = cls.batch_embed(model, (tr_Xs, tr_Ys))
                te_Xs, te_Ys = cls.batch_embed(model, (te_Xs, te_Ys))
                
#                 inner_start = time()
                # Loop over tasks. TODO: Make parallel if needed
                for j in range(len(tr_Xs)):
                    task_model = copy_model(model, grad=False)
                    output = inner_loop_iMAML(
                        task_model, (tr_Xs[j], tr_Ys[j]), (te_Xs[j], te_Ys[j]), reg_params=model_params, variant=variant,
                        grad=False, acc=not regression, regression=regression, **inner_loop_kwargs
                    )

                    # Assign an arbitrary value to acc if this is a regression task
                    if regression:
                        l = output
                        a = torch.tensor([-1.])
                    else:
                        l, a = output

                    if mean:
                        loss += l.item()
                        acc += a.item()
                        total += 1
                    else:
                        loss_list.append(l)
                        acc_list.append(a)
#                     print('Loss: {0:.4f}, Acc: {1:.4f}'.format(l.item(), a.item()))
#                 print('Eval inner time: {0:.4f}'.format(time() - inner_start))

        if mean:
            return loss / total, acc / total
        else:
            return np.array(loss_list), np.array(acc_list)


class MetaProxLearner(iMAMLLearner):
    '''
    Implements a variant of MetaProximal update based on https://papers.nips.cc/paper/2019/file/8c235f89a8143a28a1d6067e959dd858-Paper.pdf
    Note: optimizer must be SGD
    args.variant does not matter. For gradient computation both train and test data will combined
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert type(self.optimizer) == optim.SGD


    def step(self, batch):
#         start = time()
        args = self.args
        optimizer = self.optimizer
        model = self.model

        tr_Xs, tr_Ys = batch['train']
        te_Xs, te_Ys = batch['test']
        tr_Xs, tr_Ys = self.batch_embed(model, (tr_Xs, tr_Ys))
        te_Xs, te_Ys = self.batch_embed(model, (te_Xs, te_Ys))

        # Iterate over all tasks
        optimizer.zero_grad()
        model.train()
        grad_mean = 0.

        model_params = params_to_vec(model.parameters()).detach().clone()
#         inner_start = time()
        if not args.parallel:
            for i in range(len(tr_Xs)):
                task_model = copy_model(model, grad=False)
                model_grad = inner_loop_MetaProx(
                    task_model, (tr_Xs[i], tr_Ys[i]), (te_Xs[i], te_Ys[i]),
                    variant=args.variant, grad=True, acc=False, reg=args.reg,
                    regression=self.regression, reg_params=model_params, inner_lr=args.inner_lr,
                    inner_steps=args.inner_steps
                )
                grad_mean += model_grad / len(tr_Xs)

        grad_mean = vec_to_params(model, grad_mean)
        
        optimizer.zero_grad()
        apply_grad(model, grad_mean)
#         clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()
#         print('Step time: {0:.4f}'.format(time() - start))
