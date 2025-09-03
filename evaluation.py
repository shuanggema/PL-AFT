import torch
import numpy as np
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.util import Surv

def eval_linear_coef(coef_true, coef_est):
    """
    Evaluation function for linear coefficients (used in Model.py)
    @param coef_true: true coefficients
    @param coef_est: estimated coefficients
    @return: evaluation results
    """
    if isinstance(coef_true, torch.Tensor):
        coef_true = coef_true.detach().numpy()  
    if isinstance(coef_est, torch.Tensor):
        coef_est = coef_est.detach().numpy() 
    res = {}
    res["TP"] = ((coef_est!=0)*(coef_true!=0)).sum()
    res["FP"] = ((coef_est!=0)*(coef_true==0)).sum()
    res["TPR"] = res["TP"] / (coef_true!=0).sum()
    res["FPR"] = res["FP"] / (coef_true==0).sum()
    res["SSE"] = np.sqrt(((coef_true - coef_est)**2).sum()) # estimation error for coef
    return res  


def eval_nonlinear(coef_g_est, coef_g_true, 
                   y2_pred=None, y2_true=None
                   ):
    """
    Evaluation function for nonlinear coefficients (used in Model.py)
    @param coef_g_est: variable selection results for nonlinear coefficients, 0 for not selected, nonzero for selected
    @param coef_g_true: variable selection results for nonlinear coefficients, 0 for not selected, nonzero for selected
    @param y2_pred: predicted value by nonlinear part
    @param y2_true: true value for nonlinear part

    @return: evaluation results
    """
    if isinstance(coef_g_est, torch.Tensor):
        coef_g_est = coef_g_est.detach().numpy()  
    res = {}  
    res["TP_g"] = ((coef_g_est!=0)*(coef_g_true!=0)).sum()  
    res["FP_g"] = ((coef_g_est!=0)*(coef_g_true==0)).sum()  
    res["TPR_g"] = res["TP_g"] / (coef_g_true!=0).sum()  
    res["FPR_g"] = res["FP_g"] / (coef_g_true==0).sum()  
    if (y2_pred is not None) and (y2_true is not None):
        if isinstance(y2_pred, torch.Tensor):
            y2_pred = y2_pred.detach().numpy()  
        if isinstance(y2_true, torch.Tensor):
            y2_true = y2_true.detach().numpy()      
        res["SSE_g"] = np.sqrt(((y2_pred - y2_true)**2).mean()) 

    return res 


def evaluation_AUC(tra_Y_dic, test_Y_dic, risk_scores, time_points=None, **kwargs):
    """
    Evaluation function for time-dependent AUC (used in Model.py)

    @param tra_Y_dic: dictionary for training data, with keys "Y" and "N_"
    @param test_Y_dic: dictionary for testing data, with keys "Y" and "N_"
    @param risk_scores: predicted risk scores for testing data
    @param time_points: time points for calculating AUC, if None, use quantiles of observed event times in the combined data

    @return: evaluation results
    """
    res = {}
    # information of test data 
    Y_tes = test_Y_dic["Y"].cpu().numpy()
    Y_tes = np.exp(Y_tes)
    N_tes = test_Y_dic["N_"].cpu().numpy()
    y_tes = Surv.from_arrays(event=N_tes.astype(bool), time=Y_tes)
    # information of training data
    Y_train = tra_Y_dic["Y"].cpu().numpy()
    Y_train = np.exp(Y_train)
    N_train = tra_Y_dic["N_"].cpu().numpy()
    # construct survival object for KM estimator  
    Y_KM = np.concatenate([Y_train, Y_tes])
    N_KM = np.concatenate([N_train, N_tes])
    y_km = Surv.from_arrays(event=N_KM.astype(bool), time=Y_KM) 
    if time_points is None:
        time_points = np.quantile(Y_KM[N_KM == 1], [0.25, 0.5, 0.75])
    aucs, mean_auc = cumulative_dynamic_auc(y_km, y_tes, risk_scores, time_points) 
    res["time_points"] = time_points
    res["aucs"] = aucs   # AUC values at the time points
    res["mean_auc"] = mean_auc # mean AUC over the time points
    return res  
