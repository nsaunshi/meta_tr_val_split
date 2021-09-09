from copy import deepcopy
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import LogisticRegression as LogitReg
from sklearn.linear_model import Ridge
from sklearn.metrics import log_loss

from time import time
import torch
from torch import nn, optim

from utils import apply_grad, flatten, get_device, get_obj, vec_to_params, params_to_vec



DEBUG = False


##################################################
###### Representation learning inner loop ########
##################################################


def inner_loop_RepLearn(model, tr_data, te_data, variant='tr-tr', firstorder=True, grad=False, acc=False, mean=True, regression=False,
                        reg=1., custom=False, inner_lr=0.1, inner_steps=100, device='cuda:0', transductive=True, use_explicit=False):
    '''
    Run inner loop logistic regression and compute the loss, either tr-tr (if te_data is None) or tr-te
    
    model: nn.Module to map inputs to representations
    tr_data: Tuple of task train inputs and labels.
    te_data: Tuple of task test inputs and labels.
    variant: 'tr-tr' or 'tr-te'
    grad: Whether gradient needs to be stored (True for meta-training phase)
    acc: Bool. If True, then 0-1 accuracy is also computed in addition to loss.
    mean: Bool. Whether to return mean of task losses/accuracies or an array
    regression: Bool. True for regression, False for classification
    transductive: Bool. Meta-evaluation setting
    kwargs: Involves reg, custom, inner_steps, device
    '''

    if grad:
        acc = False
        mean = True

    # If meta-training without firstorder approximation, please use few inner steps
    if grad and (not firstorder):
        assert(custom)
        # assert(inner_steps < 20)

    # combine tr and te data if tr-tr variant
    if variant == 'tr-tr':
        tr_Xs, tr_Ys = tr_data
        te_Xs, te_Ys = te_data
        all_Xs = torch.stack([torch.cat([tr_Xs[i], te_Xs[i]]) for i in range(len(tr_Xs))])
        all_Ys = torch.stack([torch.cat([tr_Ys[i], te_Ys[i]]) for i in range(len(tr_Ys))])
        tr_data = (all_Xs, all_Ys)
        te_data = (all_Xs, all_Ys)


    # Get train representations
    if not grad or firstorder:
        with torch.no_grad():
            tr_Xs, tr_Ys = tr_data
            tr_reps = model(tr_Xs).detach()
    else:
        tr_Xs, tr_Ys = tr_data
        tr_reps = model(tr_Xs)


    # Solve inner optimization for all tasks.
    # Only passes the classifier layer, since representation is fixed
    start = time()
    if regression:
        Ws, bs = inner_loop_linreg(
            (tr_reps, tr_Ys), init_clf=model.classifier, reg=reg, firstorder=firstorder,
            custom=custom, inner_lr=inner_lr, inner_steps=inner_steps, device=device
        )
    else:
        Ws, bs = inner_loop_logreg(
            (tr_reps, tr_Ys), init_clf=model.classifier, reg=reg, firstorder=firstorder,
            custom=custom, inner_lr=inner_lr, inner_steps=inner_steps, device=device
        )
    # if not next(model.parameters()).grad is None:
    #     print(grad, torch.norm(next(model.parameters()).grad))
    # else:
    #     print(grad, None)
    if DEBUG:
        print('Per task time: {0:.4f}'.format((time() - start) / tr_Xs.shape[0]))


    # Get test representation. Computation depends on whether it is transductive setting or not
    # Note that te_Xs will have shape (n_tasks, n_samples, rep_dim)
    te_Xs, te_Ys = te_data
    n_tasks = te_Xs.shape[0]
    if variant == 'tr-tr':
        transductive = True

    if grad:
        te_reps = model(te_Xs)
    else:
        with torch.no_grad():
            model.train()
            # Evaluation method based on transductive or non-transductive mode
            if transductive:
                te_reps = model(te_Xs).detach()
            else:
                te_reps = []
                for i in range(n_tasks):
                    te_reps_i = []
                    for X in te_Xs[i]:
                        tr_Xs_X = torch.cat([tr_Xs[i], X.unsqueeze(0)])
                        te_reps_i.append(model(tr_Xs_X)[-1].detach())
                    te_reps.append(torch.stack(te_reps_i))
                te_reps = torch.stack(te_reps)

    
    # Compute preds for all tasks and collapse all tasks into 1 tensor
    te_preds = torch.bmm(te_reps, Ws) + bs[:,None,:]
    te_preds = flatten(te_preds)
    if not model.classifier is None:
        te_preds += model.classifier(flatten(te_reps))
    te_Ys = flatten(te_Ys)


    # Compute loss (and accuracy optionally)
    if mean:
        obj = get_obj(regression, reduction='mean')
    else:
        obj = get_obj(regression, reduction='none')

    loss = obj(te_preds, te_Ys)


    if use_explicit:
         tss = te_reps.shape[0]
         nss = te_reps.shape[1]
         loss += .001*torch.square(torch.norm(te_reps))/(tss*nss)

    if not mean:
        assert loss.shape[0] % n_tasks == 0
        loss = loss.view(n_tasks, loss.shape[0]//n_tasks).mean(dim=1)

    if acc:
        with torch.no_grad():
            accuracy = (te_preds.argmax(dim=1) == te_Ys).float()

            if not mean:
                assert accuracy.shape[0] % n_tasks == 0
                accuracy = accuracy.view(n_tasks, accuracy.shape[0]//n_tasks).mean(dim=1)
            else:
                accuracy = accuracy.mean()
        return loss, accuracy
    else:
        return loss




class ManyLinearClf(nn.Module):
    def __init__(self, n_tasks, inp_dim, out_dim):
        super(ManyLinearClf, self).__init__()
        self.Ws = nn.Parameter(torch.zeros((n_tasks, inp_dim, out_dim), dtype=torch.float), requires_grad=True)
        self.bs = nn.Parameter(torch.zeros((n_tasks, out_dim), dtype=torch.float), requires_grad=True)

    def forward(self, x):
        return torch.bmm(x, self.Ws) + self.bs[:,None,:]




def inner_loop_logreg(tr_data, init_clf=None, reg=1., firstorder=True, custom=False, inner_lr=0.1, inner_steps=100, device='cuda:0'):
    '''
    Train a inner loop linear classifier given fixed representations using logistic regression
        with l2 regularization penalizing distance from given initialzation.
    If standard l2 regularization, then there is an option of using the existing scikit implementation
        of logistic regression.

    (X_rep, Y): X_rep is torch tensor of representations with shape (n_tasks, n_samples, rep_dim).
                Y is torch tensor of labels with shape (n_tasks, n_samples, label_dim).
    init_clf: Fix classifier to use distance regularization (and as initialization for custom optimization).
    reg: Float denoting the regularization strength.
    custom: Bool denoting whether to use custom solver for inner problem or default scikit solver.
    inner_steps: Number of steps of inner loop.
    '''
    
    assert (init_clf is None) or custom, 'custom must be True if init_clf is not None'

    X_reps, Ys = tr_data
    device = X_reps.device
    n_tasks = len(X_reps)
    assert(len(X_reps) == len(Ys))

    # Use existing scikit implementations to inner loop accurately
    if not custom:
        loss = 0.
        Ws, bs = [], []
        if inner_steps == 0:
            inner_steps = 100
        for i in range(n_tasks):
            X_rep = X_reps[i]
            Y = Ys[i]
            n = X_rep.shape[0]
            X_rep_np = X_rep.detach().cpu().numpy()
            Y_np = Y.detach().cpu().numpy()
            # Refer to scikit LogisticRegression documentation for conversion from C to reg
            # https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
            logreg = LogitReg(fit_intercept=True, C=1./reg/n, multi_class='multinomial', max_iter=inner_steps)
            logreg.fit(X_rep_np, Y_np)
            Ws.append(torch.tensor(logreg.coef_).float().t().to(device))
            bs.append(torch.tensor(logreg.intercept_).to(device))
            loss += log_loss(Y_np, logreg.predict_proba(X_rep_np))
        if DEBUG:
            print('(non custom) Log loss: {0:.4f}'.format(loss / n_tasks))
        return torch.stack(Ws), torch.stack(bs)
    
    # Use solution from custom optimization of logistic regression using SGD
    if custom:
        with torch.enable_grad():
            # Variable for inner loop linear classifier. Initialize with init_clf if provided, otherwise 0
            rep_dim = X_reps.shape[-1]
            label_dim = len(torch.unique(Ys[0]))
            manylinclfs = ManyLinearClf(n_tasks, rep_dim, label_dim).to(device)

            if not init_clf is None:
                init_weight = init_clf.weight.detach().t()
            else:
                init_weight = torch.zeros((rep_dim, label_dim), dtype=torch.float).to(device)

            # Train inner loop classifier using GD with weight_decay=reg
            Ys_flat = flatten(Ys)
            # sgd_optimizer = optim.SGD(manylinclfs.parameters(), lr=inner_lr)
            sgd_optimizer = optim.SGD(manylinclfs.parameters(), lr=inner_lr, momentum=0.9)
            xent = nn.CrossEntropyLoss(reduction='mean')
            manylinclfs.train()
            for _ in range(inner_steps):
                sgd_optimizer.zero_grad()
                Y_logits = flatten(manylinclfs(X_reps))
                # print(Y_logits.shape, Ys_flat.shape, manylinclfs.Ws.shape, init_weight.shape, reg)
                loss = xent(Y_logits, Ys_flat) + 0.5 * reg * torch.square(manylinclfs.Ws - init_weight[None,:]).sum() / n_tasks
                loss.backward(retain_graph=not firstorder)
                # print('\t{0:.4f}'.format(loss.item()))
                sgd_optimizer.step()
            if DEBUG:
                with torch.no_grad():
                    print('(custom) Log loss: {0:.4f}'.format(xent(Y_logits, Ys_flat).item()))
        return manylinclfs.Ws.detach(), manylinclfs.bs.detach()




def inner_loop_linreg(tr_data, init_clf=None, reg=1., firstorder=True, custom=False, inner_lr=0.1, inner_steps=100, device='cuda:0'):
    '''
    Train a inner loop linear classifier given fixed representations using ridge regression
        with l2 regularization penalizing distance from given initialzation.
    If standard l2 regularization, then there is an option of using the existing scikit implementation
        of ridge regression.

    X_rep: Torch tensor of inputs with shape (n_tasks, n_samples, rep_dim).
    Y: Torch tensor of labels with shape (n_tasks, n_samples, label_dim).
    init_clf: Fix classifier to use distance regularization (and as initialization for custom optimization).
    reg: Float denoting the regularization strength.
    custom: Bool denoting whether to use custom solver for inner problem or default scikit solver.
    inner_steps: Number of steps of inner loop.
    '''

    X_reps, Ys = tr_data
    device = X_reps.device
    n_tasks = len(X_reps)
    assert(len(X_reps) == len(Ys))

    # Use existing scikit implementations to inner loop accurately
    if not custom:
        loss = 0.
        Ws, bs = [], []
        if inner_steps == 0:
            inner_steps = 100
        for i in range(n_tasks):
            X_rep = X_reps[i]
            Y = Ys[i]
            n = X_rep.shape[0]
            # Subtact prediction from init_clf if needed
            if not init_clf is None:
                with torch.no_grad():
                    Y -= init_clf(X_rep)
            X_rep_np = X_rep.detach().cpu().numpy()
            Y_np = Y.detach().cpu().numpy()

            linreg = Ridge(fit_intercept=True, alpha=reg*n)
            linreg.fit(X_rep_np, Y_np)
            Ws.append(torch.tensor(linreg.coef_).float().t().to(device))
            bs.append(torch.tensor(linreg.intercept_).to(device))
            loss += ((linreg.predict(X_rep_np) - Y_np)**2).mean()
        if DEBUG:
            print('(non custom) MSE loss: {0:.4f}'.format(loss / n_tasks))
        return torch.stack(Ws), torch.stack(bs)
    
    # Use solution from custom optimization of ridge regression using SGD
    if custom:
        with torch.enable_grad():
            # Variable for inner loop linear classifier. Initialize with init_clf if provided, otherwise 0
            rep_dim = X_reps.shape[-1]
            label_dim = Ys.shape[-1]
            manylinclfs = ManyLinearClf(n_tasks, rep_dim, label_dim).to(device)
            # print(X_reps.shape, Ys.shape, manylinclfs.Ws.detach().shape)

            if not init_clf is None:
                init_weight = init_clf.weight.detach().t()
            else:
                init_weight = torch.zeros((rep_dim, label_dim), dtype=torch.float).to(device)

            # Train inner loop classifier using GD with weight_decay=reg
            Ys_flat = flatten(Ys)
            # sgd_optimizer = optim.SGD(manylinclfs.parameters(), lr=inner_lr)
            sgd_optimizer = optim.SGD(manylinclfs.parameters(), lr=inner_lr, momentum=0.9)
            # sgd_optimizer = optim.Adam(manylinclfs.parameters(), lr=inner_lr)
            mse = nn.MSELoss(reduction='mean')
            manylinclfs.train()
            for i in range(inner_steps):
                sgd_optimizer.zero_grad()
                Y_preds = flatten(manylinclfs(X_reps))

                # if not init_clf is None:
                #     if firstorder:
                #         with torch.no_grad():
                #             Y_preds += init_clf(flatten(X_reps))
                #     else:
                #         Y_preds += init_clf(flatten(X_reps))

                loss = mse(Y_preds, Ys_flat) + reg * torch.square(manylinclfs.Ws - init_weight[None,:]).sum() / n_tasks
                loss.backward(retain_graph=not firstorder)
                # print('\t', i, '{0:.4f}'.format(loss.item()))
                sgd_optimizer.step()
            if DEBUG:
                with torch.no_grad():
                    print('(custom) MSE loss: {0:.4f}'.format(mse(Y_preds, Ys_flat).item()))
            return manylinclfs.Ws.detach(), manylinclfs.bs.detach()



##############################################
######### iMAML learning inner loop ##########
##############################################



def inner_loop_iMAML(task_model, tr_data, te_data, variant='tr-tr', grad=False, acc=False, regression=False,
                     reg=1.0, reg_damp=10.0, reg_params=None, inner_lr=0.01, inner_steps=20, cg_steps=5, transductive=True):
    '''
    Implement implicit MAML
    
    Args:
        model: nn.Module that computes logits/predictions for inputs
        tr_data: (tr_X, tr_Y), with tr_x.shape = (n_tr_inp, inp_dim) and tr_Y.shape = (n_tr_inp,)
        te_data: (te_X, te_Y), with te_x.shape = (n_te_inp, inp_dim) and te_Y.shape = (n_te_inp,)
        variant: 'tr-tr' or 'tr-te'
        grad: bool, If True, computes and returns the iMAML gradient w.r.t. model parameters
        acc: bool, If True, "test" accuracy is returned along with "test" loss
        reg: float, regularization coefficient
        regression: bool, regression (True) v/s classification (False)
        reg_params: parameters to use in distance regularization
        inner_lr: float, learning rate to use for inner optimization
        inner_steps: int, number of steps to run inner optimization for
        cg_steps: int, number of conjugate gradient steps for inner loop computation inverse solver
    
    Return:
        if grad is True, return model_gradients
        elif acc is False, return test_loss
        else, return test_loss, test_acc
    '''

    # combine tr and te data if tr-tr variant
#     print(reg_damp)
    if variant == 'tr-tr':
        tr_Xs, tr_Ys = tr_data
        te_Xs, te_Ys = te_data
        all_Xs = torch.cat([tr_Xs, te_Xs])
        all_Ys = torch.cat([tr_Ys, te_Ys])
        tr_data = (all_Xs, all_Ys)
        te_data = (all_Xs, all_Ys)

    # Train regularized model on train data. If no reg_params provided, use current task_model parameters
    if reg_params is None:
        reg_params = params_to_vec(task_model.parameters()).detach().clone()

    from time import time
    start = time()
    task_model = inner_loop_regularized(
        task_model, tr_data, reg=reg, regression=regression,
        reg_params=reg_params, inner_lr=inner_lr, inner_steps=inner_steps
    )
    if DEBUG:
        print('inner loop time: {0:.4f}'.format(time() - start))

    # Objective function
    obj = get_obj(regression)

    # Gradient computation is required
    if grad:
        # Compute test loss
        task_model.train()
        tr_X, tr_Y = tr_data
        te_X, te_Y = te_data
        te_preds = task_model(te_X)
        te_loss = obj(te_preds, te_Y)
        tr_preds = task_model(tr_X)
        tr_loss = obj(tr_preds, tr_Y)

        parameters = list(task_model.parameters())
        tr_grad = params_to_vec(torch.autograd.grad(tr_loss, parameters, create_graph=True))
        te_grad = params_to_vec(torch.autograd.grad(te_loss, parameters))

        @torch.enable_grad()
        def Av(v):
            Hv = torch.autograd.grad(tr_grad, parameters, retain_graph=True, grad_outputs=v)
            Hv = params_to_vec(Hv).detach()
            # Scaling is different due to MSELoss being |preds - Y|^2 instead of 0.5*|preds - Y|^2
            if regression:
                return v + Hv/(reg + reg_damp)/2.
            else:
                return v + Hv/(reg + reg_damp)
            
        model_grad_vec = conjugate_gradient(te_grad, Av, cg_steps=cg_steps)
        return model_grad_vec

    # Perform evaluation if grad=False. Return test loss (and accuracy)
    if not grad:
        # For tr-tr evaluation transductive is equivalent to non-transductive
        if variant == 'tr-tr':
            transductive = True
        with torch.no_grad():
            task_model.train()
            tr_Xs, tr_Ys = tr_data
            te_Xs, te_Ys = te_data
            # Make predictions for the test set with or without using test batch statistics
            if transductive:
                te_preds = task_model(te_Xs).detach()
            else:
                te_preds = []
                for X in te_Xs:
                    tr_Xs_X = torch.cat([tr_Xs, X.unsqueeze(0)])
                    pred = task_model(tr_Xs_X)[-1].detach()
                    te_preds.append(pred)
                te_preds = torch.stack(te_preds)

            # Measure loss and accuracy (if needed)
            te_loss = obj(te_preds, te_Ys)
            if acc:
                te_acc = (te_preds.argmax(dim=1) == te_Ys).float().mean()
                return te_loss, te_acc
            return te_loss


@torch.enable_grad()
def inner_loop_regularized(task_model, data, reg=1.0, regression=False, reg_params=None, inner_lr=0.01, inner_steps=20):
    '''
    Run the inner loop that optimizes the regularized objective for inner_steps number of steps
    '''

    X, Y = data
    if reg_params is None:
        reg_params = params_to_vec(task_model.parameters()).detach().clone()

    def reg_fn():
        return torch.norm(params_to_vec(task_model.parameters()) - reg_params)**2


    # Set up inner optimizer and objective function
    optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)
    obj = get_obj(regression)

    task_model.train()
    optimizer.zero_grad()

    for i in range(inner_steps):
        preds = task_model(X)
        if regression:
            loss = obj(preds, Y) + reg * reg_fn()
        else:
            loss = obj(preds, Y) + 0.5 * reg * reg_fn()
#         print('\t', i, '{0:.4f}'.format(loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return task_model


@torch.no_grad()
def conjugate_gradient(g, Av, cg_steps=5):
    '''
    Solve for min_x |Ax - g|^2 by running the conjugate gradient method for cg_steps steps
    '''

    # x = g.detach().clone()
    x = torch.zeros_like(g)
    r = g.detach().clone() - Av(x).detach()
    p = r.detach().clone()
    for i in range(cg_steps):
        Ap = Av(p)
        alpha = (r @ r)/(p @ Ap)
        x = x + alpha * p
        r_new = r - alpha * Ap
        beta = (r_new @ r_new)/(r @ r)
        p = r_new + beta * p
        r = r_new.detach().clone()
    return x.detach()




##############################################
######## MetaProx learning inner loop ########
##############################################


def inner_loop_MetaProx(task_model, tr_data, te_data, variant='tr-tr', grad=False, acc=False, regression=False,
                        reg=1.0, reg_params=None, inner_lr=0.01, inner_steps=20, transductive=True):
    '''
    Implement MetaProximal update
    
    Args:
        model: nn.Module that computes logits/predictions for inputs
        tr_data: (tr_X, tr_Y), with tr_x.shape = (n_tr_inp, inp_dim) and tr_Y.shape = (n_tr_inp,)
        te_data: (te_X, te_Y), with te_x.shape = (n_te_inp, inp_dim) and te_Y.shape = (n_te_inp,)
        variant: 'tr-tr' or 'tr-te'
        grad: bool, If True, computes and returns the iMAML gradient w.r.t. model parameters
        acc: bool, If True, "test" accuracy is returned along with "test" loss
        reg: float, regularization coefficient
        regression: bool, regression (True) v/s classification (False)
        reg_params: parameters to use in distance regularization
        inner_lr: float, learning rate to use for inner optimization
        inner_steps: int, number of steps to run inner optimization for
    
    Return:
        if grad is True, return model_gradients
        elif acc is False, return test_loss
        else, return test_loss, test_acc
    '''

    # If training or using tr-tr variant, use everything as train data
    if grad or variant == 'tr-tr':
        tr_Xs, tr_Ys = tr_data
        te_Xs, te_Ys = te_data
        all_Xs = torch.cat([tr_Xs, te_Xs])
        all_Ys = torch.cat([tr_Ys, te_Ys])
        tr_data = (all_Xs, all_Ys)

    if variant == 'tr-tr':
        te_data = (all_Xs, all_Ys)

    # Train regularized model on train data. If no reg_params provided, use current task_model parameters
    if reg_params is None:
        reg_params = params_to_vec(task_model.parameters()).detach().clone()
#     task_model = inner_loop_regularized(
    task_model = inner_loop_regularized(
        task_model, tr_data, reg=reg, regression=regression,
        reg_params=reg_params, inner_lr=inner_lr, inner_steps=inner_steps
    )

    # If grad is required, return simple MetaProx update grad
    if grad:
#         model_grad_vec = reg * (reg_params.detach().clone() - params_to_vec(task_model.parameters()).detach().clone())
        model_grad_vec = reg_params.detach().clone() - params_to_vec(task_model.parameters()).detach().clone()
        return model_grad_vec

    # Perform evaluation if grad=False. Return test loss (and accuracy)
    if not grad:
        # For tr-tr evaluation transductive is equivalent to non-transductive
        if variant == 'tr-tr':
            transductive = True
        with torch.no_grad():
            task_model.train()
            tr_Xs, tr_Ys = tr_data
            te_Xs, te_Ys = te_data
            # Make predictions for the test set with or without using test batch statistics
            if transductive:
                te_preds = task_model(te_Xs)
            else:
                te_preds = []
                for X in te_Xs:
                    tr_Xs_X = torch.cat([tr_Xs, X.unsqueeze(0)])
                    pred = task_model(tr_Xs_X)[-1]
                    te_preds.append(pred)
                te_preds = torch.stack(te_preds)            

            # Measure loss and accuracy (if needed)
            obj = get_obj(regression)
            te_loss = obj(te_preds, te_Ys)
            if acc:
                te_acc = (te_preds.argmax(dim=1) == te_Ys).float().mean()
                return te_loss, te_acc
            return te_loss
