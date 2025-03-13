import copy
from DGP import generate_data, data_to_torch
import pandas as pd
import numpy as np
import multiprocessing as mp
from utils import *
import pickle
import os
from Model import Partial_Linear_DNN
from Trainer import Trainer
import warnings
warnings.filterwarnings("ignore")
import time

def sim_func(seed_=1, n=1000, r=100, p=100, 
                  case=0, s1=10, rho=0.5,
                  metrics = ["metric", "metric1", "metric2", "TP", "FP", "TP_g", "FP_g", "SSE", "TPR", "FPR", "TPR_g", "FPR_g"], 
                  **kwargs):
    """
    Simulation function 

    @param seed_: random seed
    @param n: number of training samples
    @param r: number of covariates for X
    @param p: number of covariates for Z
    @param case: simulation case (simulation examples)
    @param s1: number of non-zero coefficients in X
    @param rho: correlation coefficient
    @param metrics: evaluation metrics

    @return: evaluation results, best tuning parameters, and estimated coefficients for the proposed method
    """
    #### Generate data ####
    tra_data, val_data, tes_data, coef_true, coef_g_true = \
        generate_data(n=n, n_val=200, n_tes=2000, r=r, p=p,
                  case=case, s1 = s1, censor_rate=0.3,
                  coef=None, rho=rho, seed=seed_
                  )
    tra_X, tra_Y = data_to_torch(tra_data)
    val_X, val_Y = data_to_torch(val_data)
    tes_X, tes_Y = data_to_torch(tes_data)

    result = {}
    best_tunings = {}
    coef_proposed = np.zeros(r)  

    #### Method ####      
    ## Proposed method ##
    candidate_hidden_size = [[64], [128, 64]]
    best_metric = -np.inf  
    best_result_tmp = {}  
    best_tunings_tmp = {}
    coef_proposed_tmp = np.zeros(r)  
    for hidden_size in candidate_hidden_size:
        setup_seed(7)
        model = Partial_Linear_DNN(r=r, p=p,
                    X_name = "X", Z_name="Z",
                    hidden_size=hidden_size, dropout=None, batchnorm=True,
                    penalty = "lasso"
                    )
        #### Lasso-based partial linear model for adaptive weights ####
        trainer = Trainer(model)
        # candidate hyperparameters
        kwarg_loss_init = {"gamma1": 0, "gamma2": 0}
        kwarg_prox_set = {"lam": [0.02, 0.01, 0.04], 
                        "mu": [0.4, 0.1, 0.2]}  
        kwarg_prox_set = {"lam": [0.02, 0.01, 0.04], 
                        "mu": [0.05, 0.1, 0.2]}   
        kwarg_loss_set = {"gamma1": [0], "gamma2":[0]}
        kwarg_thresh = {}
        lrs = [0.008]  
        lrs_nonlinear = [0.004]
        start_time = time.time()
        # initialization
        trainer.train(tra_X, tra_Y, lr=0.008, 
                    lr_nonlinear=0.004, is_refit=True, maxit=25, 
                    kwarg_loss=kwarg_loss_init, use_lr_scheduler=False
                    )
        model_state_init = quick_deepcopy(trainer.model.state_dict())  
        computation_time_proposed_init = time.time() - start_time
        start_time = time.time()
        # training (grid search to select best hyperparameters)
        trainer.train_path(tra_X, tra_Y,
                            kwarg_prox_set=kwarg_prox_set, kwarg_loss_set=kwarg_loss_set, 
                            kwarg_thresh=kwarg_thresh,
                            lrs=lrs, lrs_nonlinear=lrs_nonlinear, 
                            prox_it=0, min_it=10, maxit=200, early_stop_round=10,
                            init_state_dict=model_state_init,
                            val_X_dic=val_X, val_Y_dic=val_Y,
                            refit=True, use_lr_scheduler=False
                            )
        trainer.model.gen_part1_coef(**tra_X, **tra_Y)
        computation_time_proposed_lasso = (time.time() - start_time)/9 + computation_time_proposed_init
        # Generate the adaptive weights
        eta, zeta = model.gen_adaptive_weights(gamma=2)  
        
        #### Proposed Adaptive Lasso ####
        model.load_state_dict(model_state_init)  
        # candidate hyperparameters
        kwarg_prox_set = {"lam": [0.02, 0.01, 0.04], 
                          "mu": [0.2, 0.1, 0.4],
                          "zeta": [zeta],
                          "eta": [eta]
                          }  
        kwarg_loss_set = {"gamma1": [0], "gamma2":[0]}
        lrs = [0.008]  
        lrs_nonlinear = [0.004]
        start_time = time.time()
        # training (grid search to select best hyperparameters)
        trainer.train_path(tra_X, tra_Y,
                            kwarg_prox_set=kwarg_prox_set, kwarg_loss_set=kwarg_loss_set, 
                            lrs=lrs, lrs_nonlinear=lrs_nonlinear,
                            prox_it=0, min_it=10, maxit=200, early_stop_round=10,
                            init_state_dict=model_state_init,
                            val_X_dic=val_X, val_Y_dic=val_Y,
                            refit=True, use_lr_scheduler=False
                            )
        trainer.model.gen_part1_coef(**tra_X, **tra_Y)
        computation_time_proposed = (time.time() - start_time)/9 + computation_time_proposed_lasso
        # model evaluation
        res = trainer.model.evaluation(tes_X_dic=tes_X, tes_Y_dic=tes_Y, 
                                        coef_true=coef_true, coef_g_true=coef_g_true
                                        )
        best_result_tmp["Proposed.Ada"] = {"Time": computation_time_proposed, **res}  
        tunings = trainer.best_tunings
        tunings = (tunings, hidden_size)
        best_tunings_tmp["Proposed.Ada"] = tunings
        coef_proposed_tmp = trainer.model.part1_layer.weight.detach().squeeze().numpy()

        if trainer.best_metric > best_metric:
            best_metric = trainer.best_metric  
            result.update(best_result_tmp) 
            best_tunings.update(best_tunings_tmp)
            coef_proposed = coef_proposed_tmp

    return pd.DataFrame(result).T[metrics], best_tunings, coef_proposed


def sim_multi_times(seed_lst, sim_func, kwargs, cores=None, folder_path=None, label=""):
    """
    Parallel run simulation multiple times

    @param seed_lst: list of random seeds
    @param sim_func: simulation function
    @param kwargs: arguments for simulation function
    @param cores: number of cores
    @param folder_path: folder path for saving results
    @param label: label for saving results
    """
    if cores is None:
        cores = min(mp.cpu_count() - 2, len(list(seed_lst)))
    print(f"{cores} cores used.")
    p = mp.Pool(cores)
    res = []
    for seed_ in seed_lst:
        kw = copy.deepcopy(kwargs)
        kw["seed_"] = seed_
        out = p.apply_async(sim_func, kwds=kw)
        res.append(out)
    p.close()
    p.join()

    res_df_lst, res_tunings_lst, res_coef_lst = list(zip(*[out.get() for out in res]))
    p.terminate()

    res_df = merge_DataFrame(list(res_df_lst))
    res_tunings = merge_dics(list(res_tunings_lst))
    res_coefs = np.stack(res_coef_lst, axis=0)

    print(res_df)

    if folder_path is not None:
        res_df.to_csv(os.path.join(folder_path, f"{label}_result.csv"))
        pickle.dump(res_tunings, open(os.path.join(folder_path, f"{label}_tunings.pkl"), "wb") )
        np.save(os.path.join(folder_path, f"{label}_res_coefs.npy"), res_coefs)

