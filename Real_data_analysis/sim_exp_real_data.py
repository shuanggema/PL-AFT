import os  
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import creater_logger, setup_seed, quick_deepcopy, merge_DataFrame
import pandas as pd  
import numpy as np 
from DGP import KM_weights, data_to_torch
from Model import Partial_Linear_DNN  
from Trainer import Trainer
import multiprocessing as mp
import copy


def real_data(seed_, X, Z, Y, N_,
              val_size=0.2, tes_size=0.2, **kwargs):
    #### Generate data ####
    p = Z.shape[1]  
    r = X.shape[1]  
    n = Z.shape[0]  

    idxs = np.arange(n)
    setup_seed(seed_)
    np.random.shuffle(idxs)

    n_val = int(n*val_size)  
    n_tes = int(n*tes_size)  
    n_tra = n - n_val - n_tes

    tra_data = dict(Z=Z[idxs[:n_tra],:], X=X[idxs[:n_tra],:], 
                    Y=Y[idxs[:n_tra]], N_=N_[idxs[:n_tra]],
                    W = KM_weights(Y[idxs[:n_tra]], 
                                   N_[idxs[:n_tra]])
                    )
    val_data = dict(Z=Z[idxs[n_tra:(n_tra+n_val)],:], X=X[idxs[n_tra:(n_tra+n_val)],:], 
                    Y=Y[idxs[n_tra:(n_tra+n_val)]], N_=N_[idxs[n_tra:(n_tra+n_val)]],  
                    W = KM_weights(Y[idxs[n_tra:(n_tra+n_val)]], 
                                   N_[idxs[n_tra:(n_tra+n_val)]])
                    )
    intercept = (tra_data["Y"]*tra_data["W"]).sum()/tra_data["W"].sum()
    tra_data["Y"] = tra_data["Y"] - intercept  
    val_data["Y"] = val_data["Y"] - intercept  

    tra_X, tra_Y = data_to_torch(tra_data)
    val_X, val_Y = data_to_torch(val_data)

    if tes_size > 0:
        tes_data = dict(Z=Z[idxs[(n_tra+n_val):],:], X=X[idxs[(n_tra+n_val):],:], 
                        Y=Y[idxs[(n_tra+n_val):]], N_=N_[idxs[(n_tra+n_val):]],
                        W = KM_weights(Y[idxs[(n_tra+n_val):]], 
                                       N_[idxs[(n_tra+n_val):]])
                        )
        tes_data["Y"] = tes_data["Y"] - intercept  
        tes_X, tes_Y = data_to_torch(tes_data)

    result = {}
    coefs_linear = np.zeros(r)  
    coefs_nonlinear = np.zeros(p)  

    #### Method ####  
    ## Proposed method ##
    candidate_hidden_size = [[32], [64], [128, 64]]  #[[128, 64]]
    best_metric = -np.inf  
    best_result_tmp = {}  
    best_result_auc_tmp = {}
    best_tunings_tmp = {}
    coefs_linear_tmp = np.zeros((r, 2))  
    coefs_nonlinear_tmp = np.zeros((p, 2))  
    for hidden_size in candidate_hidden_size:
        setup_seed(402)
        model = Partial_Linear_DNN(r=r, p=p,
                    X_name = "X", Z_name="Z",
                    hidden_size=hidden_size, dropout=None, batchnorm=True,
                    penalty = "lasso"
                    )
        #model.load_state_dict(model_state_init)
        # init coef for \beta  
        #init_coef = torch.from_numpy(coefs_linear[:,0]).to(torch.float32).unsqueeze(0)
        #model.part1_layer.weight.data = init_coef

        ## Proposed Lasso ## 
        trainer = Trainer(model)
        kwarg_loss_init = {"gamma1": 0, "gamma2": 0}
        kwarg_prox_set = {"lam": [0.005, 0.01, 0.02], 
                        "mu": [0.005, 0.01, 0.02]}  
        kwarg_loss_set = {"gamma1": [0], "gamma2":[0]}
        kwarg_thresh = {}
        lrs = [0.008]  
        lrs_nonlinear = [0.008]

        trainer.train(tra_X, tra_Y, lr=0.008, 
                   lr_nonlinear=0.008, is_refit=True, maxit=20, 
                   kwarg_loss=kwarg_loss_init, use_lr_scheduler=False
                   )
        model_state_init = quick_deepcopy(trainer.model.state_dict())  

        trainer.train_path(tra_X, tra_Y,
                            kwarg_prox_set=kwarg_prox_set, kwarg_loss_set=kwarg_loss_set, 
                            kwarg_thresh=kwarg_thresh,
                            lrs=lrs, lrs_nonlinear=lrs_nonlinear, 
                            prox_it=0, min_it=0, maxit=100, early_stop_round=10,
                            init_state_dict=model_state_init,
                            val_X_dic=val_X, val_Y_dic=val_Y,
                            refit=True, use_lr_scheduler=False
                            )
                
        ## Proposed Adaptive Lasso ## 
        eta, zeta = model.gen_adaptive_weights(gamma=2)  
        model.load_state_dict(model_state_init)  
        kwarg_prox_set = {"lam": [2*x for x in [0.005, 0.01, 0.02]], 
                        "mu": [2*x for x in [0.005, 0.01, 0.02]]}  # our_mcp 
        kwarg_loss_set = {"gamma1": [0.01], "gamma2":[0.01]}
        lrs = [0.008]  
        lrs_nonlinear = [0.008]  
        kwarg_prox_set["zeta"] = [zeta]  
        kwarg_prox_set["eta"] = [eta]  
        
        trainer.train_path(tra_X, tra_Y,
                            kwarg_prox_set=kwarg_prox_set, kwarg_loss_set=kwarg_loss_set, 
                            lrs=lrs, lrs_nonlinear=lrs_nonlinear,
                            prox_it=0, min_it=0, maxit=100, early_stop_round=10,
                            init_state_dict=model_state_init,
                            val_X_dic=val_X, val_Y_dic=val_Y,
                            refit=True, use_lr_scheduler=False
                            )
        res = trainer.model.evaluation(tes_X_dic=tes_X, tes_Y_dic=tes_Y, 
                                        coef_true=None, coef_g_true=None
                                        )
        best_result_tmp["Proposed.Ada"] = res  
        coefs_linear_tmp = trainer.model.part1_layer.weight.detach().squeeze().numpy()  
        coefs_nonlinear_tmp = trainer.model.part2_important.data.detach().squeeze().numpy()    

        if trainer.best_metric > best_metric:
            best_metric = trainer.best_metric  
            result.update(best_result_tmp) 
            coefs_linear = coefs_linear_tmp  
            coefs_nonlinear = coefs_nonlinear_tmp

    return pd.DataFrame(result).T[["metric1"]], coefs_linear, coefs_nonlinear


def sim_multi_times_real_data(seed_lst, real_data, kwargs, cores=None, folder_path=None, label=""):
    if cores is None:
        cores = min(mp.cpu_count() - 2, len(list(seed_lst)))
    print(f"{cores} cores used.")
    p = mp.Pool(cores)
    res = []
    for seed_ in seed_lst:
        kw = copy.deepcopy(kwargs)
        kw["seed_"] = seed_
        out = p.apply_async(real_data, kwds=kw)
        res.append(out)
    p.close()
    p.join()

    res_df_lst, coefs_linear_lst, coefs_nonlinear_lst = list(zip(*[out.get() for out in res]))
    p.terminate()

    res_df = merge_DataFrame(list(res_df_lst))
    res_coefs_linear = np.stack(coefs_linear_lst, axis=0)  
    res_coefs_nonlinear = np.stack(coefs_nonlinear_lst, axis=0)  

    if folder_path is not None:
        res_df.to_csv(os.path.join(folder_path, f"{label}_result.csv"))
        np.save(os.path.join(folder_path, f"{label}_coefs_linear.npy"), res_coefs_linear)
        np.save(os.path.join(folder_path, f"{label}_coefs_nonlinear.npy"), res_coefs_nonlinear)



if __name__ == "__main__":
    # =============================================  
    rep_times = 2
    folder = f"./Real_data_analysis/real_data_result/"      # folder for saving results
    logger_file = os.path.join(folder, "training_7705.log")  
    # =============================================  
    if not os.path.exists(folder):  
        os.mkdir(folder)  
    logger = creater_logger(logger_file)  

    # ==== load data ====  
    data_folder = "./Real_data_analysis/processed_data/"
    X_clip = pd.read_csv(os.path.join(data_folder, "X_clip.csv"), index_col=0)  # Imaging  
    Z_clip = pd.read_csv(os.path.join(data_folder, "Z_clip.csv"), index_col=0)  # Gene  
    E_clip = pd.read_csv(os.path.join(data_folder, "E_clip.csv"), index_col=0)  # Clinical features
    survival_outcome = pd.read_csv(os.path.join(data_folder, "survival_outcome.csv")) # survival outcome
    Y = np.log(survival_outcome["OS_MONTHS"].values) 
    N_ = survival_outcome["OS_STATUS"].values

    Z = np.concatenate([E_clip.values, Z_clip.values], axis=1)  # Linear part
    X = X_clip.values                                           # Nonlinear part
    Z_cols = list(E_clip.columns) + list(Z_clip.columns)
    X_cols = list(Z_clip.columns)
    print(Z.shape, X.shape)
    setting = dict(X=Z, Z=X, Y=Y, N_=N_, val_size=1/5, tes_size=1/5)

    # ==== Run the experiments ====  
    sim_multi_times_real_data(range(10086, 10086+rep_times), 
                    real_data, setting, folder_path=folder, cores=2, label="0"
                    )  

