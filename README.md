# Train-validation splitting in meta-learning
Code for the paper [1] that understands the benefit of train-validation splitting in meta-learning. If you find the code useful, please cite the following

```
@inproceedings{saunshi2021representation,
  author={Nikunj Saunshi and Arushi Gupta and Wei Hu},
  title={A Representation Learning Perspective on the Importance of Train-Validation Splitting in Meta-Learning},
  booktitle={Proceedings of the 38th International Conference on Machine Learning},
  year={2021}
}
```

## Requirements
numpy, sklearn, torch, torchvision, torchmeta, pyyaml, wandb

## Example runs
Steps to run RepLearn (Algorithm 1 in [1]) on MiniImageNet using the tr-val and tr-tr objectives and evaluating 5-way 1-shot test accuracy
```
cd src/

# Train
python meta_learn.py ../configs/mini_5w1s_tr_te_reg0.0_cus_lr0.05_is100_replearn_usecnn_netwidth1.0.txt
python meta_learn.py ../configs/mini_5w1s_tr_tr_reg0.0_cus_lr0.05_is100_replearn_usecnn_netwidth1.0.txt

# Evaluate
python measure_accuracy.py ../configs/mini_5w1s_tr_te_reg0.0_cus_lr0.05_is100_replearn_usecnn_netwidth1.0.txt ../models/mini_5w1s_tr_te_reg0.0_cus_lr0.05_is100_replearn_usecnn_netwidth1.0/checkpoint-30000-final.ckpt
python measure_accuracy.py ../configs/mini_5w1s_tr_tr_reg0.0_cus_lr0.05_is100_replearn_usecnn_netwidth1.0.txt ../models/mini_5w1s_tr_tr_reg0.0_cus_lr0.05_is100_replearn_usecnn_netwidth1.0/checkpoint-30000-final.ckpt
```
Can use fully connected network instead of convolutional network by replacing using configs/mini_5w1s_tr_te_reg0.0_cus_lr0.05_is100_replearn_usefc_netwidth1.0.txt instead.

Steps to run iMAML [2] using this [code](https://github.com/aravindr93/imaml_dev) on Omniglot with tr-val and tr-tr objectives
```
# Before use on iMAML one will have to generate task definitions
cd imaml_orig_src/imaml_dev-master/
python generate_task_defs.py --save_dir ./task_defs --N_way 5 --K_shot 1 --num_tasks 5000 --data_dir $DATA_DIR

# Train with tr-tr objective as follows. Add --tr_val flag to run with the usual tr-val objective
python examples/omniglot_implicit_maml.py --save_dir 20_way_1_shot_lam1.0 --N_way 20 --K_shot 1 --inner_lr 1e-1 --outer_lr 1e-3 --n_steps 25 --meta_steps 30000 --num_tasks 300000 --task_mb_size 32 --lam 1.0 --cg_steps 5 --cg_damping 1.0 --load_tasks $DATA_DIR/Omniglot_5_way_1_shot.pickle

# Evaluate iMAML model
python examples/measure_accuracy2.py --load_agent 20_way_1_shot_lam1.0/final_model.pickle --N_way 20 --K_shot 1 --num_tasks 600 --n_steps 16 --lam 0.0 --inner_lr 1e-1 --task Omniglot
```


## References
[1] Saunshi et al., A Representation Learning Perspective on the Importance of Train-Validation Splitting in Meta-Learning, *ICML* 2021
[2] Rajeswaran et al., Meta-Learning with Implicit Gradients, *NeurIPS* 2019


