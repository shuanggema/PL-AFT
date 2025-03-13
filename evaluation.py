import torch
import numpy as np


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


