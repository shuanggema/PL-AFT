import sys
sys.path.append("Z:/python_packages")
import torch
from torch.optim import Adam, AdamW, SGD
import numpy as np
from utils import *
from itertools import product
from torch.optim.lr_scheduler import StepLR


class Trainer:
    """
    Trainer for the model
    """
    def __init__(self, model):
        """
        @param model: model to estimate (class Partial_Linear_DNN in `Model.py`)
        """
        self.model = model
        self.best_tunings = None
        self.best_metric = -np.inf
        self.path = dict()

    def train(self, tra_X_dic, tra_Y_dic,
              kwarg_prox={}, kwarg_loss={}, lr=0.5, lr_nonlinear=None, maxit=200, init_state_dict=None,
              val_X_dic=None, val_Y_dic=None,
              prox_it=5, min_it=20, eval_it=2, early_stop_round = 10,
              is_refit = False, use_lr_scheduler=False
              ):
        """
        Train the model under the given hyperparameters
        @param tra_X_dic: training data for X (covariates, containing X and Z)
        @param tra_Y_dic: training data for Y (responses)
        @param kwarg_loss: hyperparameters for loss function
        @param kwarg_prox: hyperparameters for proximal term
        @param lr: learning rate for linear part
        @param lr_nonlinear: learning rate for nonlinear part
        @param maxit: maximum iteration
        @param init_state_dict: initial state of the model
        @param val_X_dic: validation data for X
        @param val_Y_dic: validation data for Y
        @param prox_it: iteration to start proximal term
        @param min_it: minimum iteration to stop training
        @param eval_it: iteration to evaluate validation data
        @param early_stop_round: early stopping round (stop training if no improvement in this number of rounds)
        @param is_refit: whether the refit is conducted
        @param use_lr_scheduler: whether to use learning rate scheduler

        @return: hyperparameters, and evaluation metrics
        """
        if lr_nonlinear is None:
            lr_nonlinear = lr

        model = self.model
        if init_state_dict:
            model.load_state_dict(init_state_dict)

        # optimizer
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for name_, p in param_optimizer if ("part1" in name_) and ("bias" not in name_)],   
            "lr": lr, "momentum":0.95},
            {'params': [p for name_, p in param_optimizer if ("part1" in name_) and ("bias" in name_)],   
            "lr": lr, "momentum":0.95},
            {'params': [p for name_, p in param_optimizer if "part2" in name_], 
            "lr": lr_nonlinear, "momentum":0.95},
            {'params': [p for name_, p in param_optimizer if "batchnorm" in name_], 
            "lr": 0.1, "momentum":0.9},
        ]
        optimizer = SGD(optimizer_grouped_parameters)

        # lr scheduler
        if use_lr_scheduler:
            scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

        # Training iteration
        val_metrics = []
        best_metric, best_it, best_state = -np.inf, 0, None  
        for i in range(maxit):
            model.train()
            optimizer.zero_grad()
            lr_cur = optimizer.param_groups[0]["lr"]
            kwarg_loss["lr"] = lr_cur
            kwarg_prox["lr"] = lr_cur

            out = model(**tra_X_dic)
            loss = model.loss_func(out, **tra_Y_dic)   # weighted least square loss
            if (not is_refit) and i > prox_it:  
                loss += model.prox_loss(**kwarg_prox)  # loss for penalty terms
            loss += model.l2_regularization(**kwarg_loss) # loss for l2 regularization
            loss.backward()
            optimizer.step()

            if use_lr_scheduler:
                scheduler.step()

            with torch.no_grad():
                # For refitting, setting the non-important coefficients to zero
                if is_refit:
                    model.refit_prox()  
                
                # Evaluate the model on validation data   
                if i>min_it and i%eval_it==0 and val_X_dic is not None:
                    metric = self.model.evaluation(tes_X_dic=val_X_dic, tes_Y_dic=val_Y_dic)["metric"]
                    val_metrics.append(metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_it = i
                        best_state = quick_deepcopy(model.state_dict())
                    else:
                        if i - best_it > early_stop_round:
                            break

        if val_X_dic is not None and best_state is not None:
            model.load_state_dict(best_state)
        else:
            best_it = i
        return (kwarg_prox, kwarg_loss, lr, lr_nonlinear, best_it), best_metric

    def train_path(self, tra_X_dic, tra_Y_dic,
                   kwarg_prox_set={}, kwarg_loss_set={}, kwarg_thresh={},
                   maxit=30, lrs=[0.1, 0.5], lrs_nonlinear=[None],
                   init_state_dict=None,
                   val_X_dic={}, val_Y_dic={},
                   prox_it=5, min_it=6, eval_it=5, early_stop_round=10,
                   refit = False, use_lr_scheduler=False
                   ):
        """
        Train the model under different hyperparameters (grid search), and select the best model 
        The best model is updated to the model attribute

        @param tra_X_dic: training data for X (covariates, containing X and Z)
        @param tra_Y_dic: training data for Y (responses)
        @param kwarg_prox_set: hyperparameters for proximal term
        @param kwarg_loss_set: hyperparameters for loss function
        @param kwarg_thresh: threshold for determining importance
        @param maxit: maximum iteration
        @param lrs: learning rates for linear part
        @param lrs_nonlinear: learning rates for nonlinear part
        @param init_state_dict: initial state of the model
        @param val_X_dic: validation data for X
        @param val_Y_dic: validation data for Y
        @param prox_it: iteration to start proximal term
        @param min_it: minimum iteration to stop training
        @param eval_it: iteration to evaluate validation data
        @param early_stop_round: early stopping round (stop training if no improvement in this number of rounds)
        @param refit: whether conduct refit after the model training
        @param use_lr_scheduler: whether to use learning rate scheduler
        """
        best_metric = -np.inf
        best_model = None
        best_tunings = None
        if init_state_dict is None:
            init_state_dict = quick_deepcopy(self.model.state_dict())

        names_prox, cand_prox = zip(*list(kwarg_prox_set.items()))
        names_loss, cand_loss = zip(*list(kwarg_loss_set.items()))

        i = 0
        for x, lst_prox in enumerate(product(*cand_prox)):
            for y, lst_loss in enumerate(product(*cand_loss)):
                for z, lr in enumerate(lrs):
                    for w, lr_nonlinear in enumerate(lrs_nonlinear):
                        kwarg_prox = {name: val for name, val in zip(names_prox, lst_prox)}
                        kwarg_loss = {name: val for name, val in zip(names_loss, lst_loss)}

                        # Train the model under this specific hyperparameters
                        tunings, metric = self.train(tra_X_dic, tra_Y_dic,
                                kwarg_prox=kwarg_prox, kwarg_loss=kwarg_loss, 
                                lr=lr, lr_nonlinear=lr_nonlinear,
                                maxit=maxit, init_state_dict=init_state_dict,
                                val_X_dic=val_X_dic, val_Y_dic=val_Y_dic,
                                prox_it=prox_it, min_it=min_it, eval_it=eval_it, early_stop_round=early_stop_round,
                                use_lr_scheduler=use_lr_scheduler
                                )
                        
                        # Determine the importance of the coefficients
                        self.model.determine_importance(**kwarg_thresh)
                        self.model.refit_prox()

                        # Refit the model
                        if refit:
                            # refit
                            part1_important = self.model.part1_important.data.clone()  
                            part2_important = self.model.part2_important.data.clone()
                            self.model.load_state_dict(init_state_dict)  
                            self.model.part1_important.data = part1_important
                            self.model.part2_important.data = part2_important
                            _, metric = self.train(tra_X_dic, tra_Y_dic,
                                                        kwarg_prox={}, kwarg_loss={}, 
                                                        lr=lr, lr_nonlinear=lr_nonlinear,
                                                        maxit=1000, init_state_dict=None,
                                                        val_X_dic=val_X_dic, val_Y_dic=val_Y_dic,
                                                        prox_it=0, min_it=20, eval_it=eval_it, 
                                                        early_stop_round=early_stop_round, is_refit=True
                                                        )

                        # Save the results
                        self.path[(tuple(kwarg_prox.items()), tuple(kwarg_loss.items()))] = metric
                        p1, p2 = self.model.num_nonzero()  
                        if i == 0 or (p1>0 and p2>0 and p1<100 and p2<100 and metric > best_metric):
                            best_metric = metric
                            best_model = quick_deepcopy(self.model.state_dict())
                            best_tunings = tunings
                        i += 1

        self.model.load_state_dict(best_model)
        self.best_tunings = best_tunings
        self.best_metric = best_metric
