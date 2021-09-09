import os
import sys
from time import time
from torch import optim
import yaml

from learners import RepLearner, iMAMLLearner, MetaProxLearner
from data_loaders import sine_loaders, sine_kshotNway, simul_loaders, simul_kshotNway
from data_loaders import omniglot_loaders, omniglot_kshotNway, miniimagenet_loaders, miniimagenet_kshotNway
from models import SineModel, LinearRepModel, OmniglotModel, MiniImagenetModel
from utils import DotDict, get_device, mkdir, get_latest_checkpoint, set_seed


WANDB_PROJECT = 'meta_split'
CHECKPOINT_DIR = '../models/'



def set_args(args):
    if args.learner is None:
        sys.exit('learner needs to be set to either replearn or imaml')

    if args.save is None:
        args.save = False

    if args.classifier is None:
        args.classifier = False

    if args.custom is None:
        args.custom = True

    if args.firstorder is None:
        args.firstorder = True

    if args.train_inner_steps is None and args.eval_inner_steps is None:
        if args.inner_steps is None:
            sys.exit('Either inner_steps or train_inner_steps and eval_inner_steps need to be set')
        if args.debug:
            print('Set train_inner_steps and eval_inner_steps to', args.inner_steps)
        args.train_inner_steps = args.inner_steps
        args.eval_inner_steps = args.inner_steps
    elif args.eval_inner_steps is None:
        args.eval_inner_steps = args.train_inner_steps

    if args.transductive is None:
        args.transductive = True

    if args.wandb:
        if args.wandb_project is None:
            args.wandb_project = WANDB_PROJECT

    if args.save:
        if args.ckpt_dir is None:
            args.ckpt_dir = CHECKPOINT_DIR

    return args



def main():
    config = yaml.load(open(sys.argv[1], 'r'), Loader=yaml.FullLoader)
    args = DotDict(config)
    args = set_args(args)

    if args.prefix:
        run_name = args.prefix
    else:
        run_name = os.path.basename(sys.argv[1])[:-4]
    print(run_name)


    ## Set up weights and biases if needed 
    if args.wandb:
        if args.debug:
            print('Setting up weights and biases')
        import wandb
        wandb.init(project=args.wandb_project, config=config)
        wandb.run.name = run_name


    ## Set appropriate random seeds
    set_seed(args)


    ## Do we need model to be end-to-end?
    if args.learner.lower() == 'replearn':
        e2e = False
    else:
        e2e = True


    ## Get meta dataset loaders and model
    if args.dataset.lower() == 'sine':
        loaders = sine_loaders(args)
        model = SineModel(args.hid_dim, e2e=e2e, classifier=args.classifier)
        regression = True

    if args.dataset.lower() == 'simul':
        loaders = simul_loaders(args)
        model = LinearRepModel(args.inp_dim, args.inp_dim, e2e=e2e, classifier=args.classifier)
        regression = True

    if args.dataset.lower() == 'omni' or args.dataset.lower() == 'omniglot':
        if args.usefc:
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


    ## Set optimizer and learner
    if args.learner.lower() == 'replearn':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        learner = RepLearner(args, model, optimizer, regression=regression)

    if args.learner.lower() == 'imaml':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        learner = iMAMLLearner(args, model, optimizer, regression=regression)

    if args.learner.lower() == 'metaprox':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        learner = MetaProxLearner(args, model, optimizer, regression=regression)

    learner.to(get_device(args.device))


    ## Evaluate existing checkpoint and finish run
    if args.evaluate_ckpt:
        latest_checkpoint = get_latest_checkpoint(args.evaluate_ckpt)
        learner.load_checkpoint(latest_checkpoint)
        learner.log(learner.evaluate([('te', False, loaders[2]), ('val', False, loaders[1])]))
        return


    ## Load checkpoint to resume training if needed
    if args.resume_ckpt:
        latest_checkpoint = get_latest_checkpoint(args.resume_ckpt)
        learner.load_checkpoint(latest_checkpoint)


    ## Create checkpointing directory if needed
    if args.save:
        args.save_dir = os.path.join(args.ckpt_dir, run_name, '')
        mkdir(args.save_dir)


    ####### Begin training
    learner.train()
    iteration = 0
    if args.debug:
        print('Running \'{0}\' method on device \'{1}\' for \'{2}\' variant'.format(args.learner, args.device, args.variant))

    for e in range(args.epochs):
        for batch in loaders[0]:
            # End if sufficient number of iterations are reached
            if iteration >= args.iterations:
                break

            # Checkpoint model if needed
            if args.save and iteration % args.save_freq == 0:
                learner.checkpoint(iteration, final=False)

            # Evaluate model if needed
            if args.eval_freq > 0 and iteration % args.eval_freq == 0:
                if regression:
                    loaders_info = [('te', False, loaders[-1]), ('va', False, loaders[1])]
                else:
                    loaders_info = [('te', False, loaders[-1])]
                learner.zero_grad()
                start = time()
                log_data = learner.evaluate(loaders_info, transductive=args.transductive)
                print('eval time: {0:.4f} s'.format(time() - start))
                learner.log(log_data, iteration)

            learner.zero_grad()
            # start = time()
            learner.step(batch)
            # print('step: {0:.4f}'.format(time() - start))
            iteration += 1

    learner.zero_grad()
    # Save model before we forget
    if args.save:
        learner.checkpoint(iteration, final=True)

    # Final eval at end of epoch
    if args.eval_freq > 0:
        loaders_info = [('te', False, loaders[-1])]
        log_data = learner.evaluate(loaders_info, transductive=args.transductive)
        learner.log(log_data, iteration)
    ####### End training


    # End WandB session
    if args.wandb:
        wandb.finish()



if __name__ == '__main__':
    main()
