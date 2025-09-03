import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from lassonet.cox import concordance_index  
from evaluation import eval_linear_coef, eval_nonlinear, evaluation_AUC


# AFT model  
class Partial_Linear_DNN(nn.Module): 
    """
    Partial linear DNN model
    """
    def __init__(self, r=100, p=100,
                 X_name = "X", Z_name="Z",
                 hidden_size=(32, 16), dropout=None, batchnorm=True,
                 penalty = "mcp"
                 ):
        """
        @param r: number of covariates for X
        @param p: number of covariates for Z
        @param X_name: name of X (in input data dictionary)
        @param Z_name: name of Z (in input data dictionary)
        @param hidden_size: hidden layer size
        @param dropout: dropout rate
        @param batchnorm: batch normalization
        @param penalty: penalty type
        """
        super().__init__()
        self.X_name = X_name
        self.Z_name = Z_name
        self.r = r
        self.p = p
        # Parametric part
        self.part1_layer = nn.Linear(r, 1, bias=True)
        pre_norm = self.part1_layer.weight.data.abs().mean()  
        a = 0.004/pre_norm
        self.part1_layer.weight.data = self.part1_layer.weight.data * a

        # Nonparametric part
        self.dropout = nn.Dropout(p=dropout) if dropout is not None else None
        dims = [p] + list(hidden_size) + [1]
        self.part2_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 2)] + 
            [nn.Linear(dims[-2], dims[-1], bias=False)]
        )
        pre_norm_avg = ((self.part2_layers[0].weight.data)**2).sum(dim=0).sqrt().mean()
        a = 0.004/pre_norm_avg  
        self.part2_layers[0].weight.data = self.part2_layers[0].weight.data * a

        if batchnorm:
            self.layers_batchnorm = nn.ModuleList(
                [nn.BatchNorm1d(dims[i+1]) for i in range(len(dims) - 2)]
            )
        else:
            self.layers_batchnorm = [None] * (len(dims) - 2)
        self.penalty = penalty.lower()
        
        # Indicators for important variables
        self.part1_important = nn.Parameter(torch.ones(r), requires_grad=False)  
        self.part2_important = nn.Parameter(torch.ones(p), requires_grad=False)  

    def forward(self, **tra_X_dic):  
        """
        Forward pass
        @param tra_X_dic: training data dictionary
        @return: prediction
        """
        X = tra_X_dic[self.X_name]  
        Z = tra_X_dic[self.Z_name]  

        # linear part
        y1_pred = self.part1_layer(X)  
        # nonparametric part
        res = Z  
        for i, (layer_linear, layer_batchnorm) in enumerate(zip(self.part2_layers[:-1], self.layers_batchnorm)):
            res = layer_linear(res)
            if layer_batchnorm is not None:
                res = layer_batchnorm(res)
            res = F.relu(res)
            if self.dropout is not None:
                res = self.dropout(res)
        y2_pred = self.part2_layers[-1](res)
        y1_pred = y1_pred.squeeze()  
        y2_pred = y2_pred.squeeze()  
        pred = y1_pred + y2_pred  
        #pred = pred - pred.mean()
        return pred, y1_pred, y2_pred

    def loss_func(self, out, Y, W, **kwargs):
        """
        Loss function
        @param out: output from forward pass
        @param Y: observerd survival time
        @param W: weight

        @return: loss
        """
        pred = out[0] - out[0].mean()  
        loss = ((pred - Y)**2 * W).sum()
        return loss 

    def metric_func(self, out, Y, N_, W, T=None, **kwargs):
        """
        Metric function
        @param out: output from forward pass
        @param Y: observerd survival time
        @param N_: censoring indicator
        @param W: weight
        @param T: true survival time (availabel for simulations)

        @return: -WMSE: negative weighted MSE
        @return: C-index: concordance index
        @return: MSE: mean squared error
        """
        pred = out[0] - out[0].mean()
        C_index = concordance_index(-pred, Y, N_)  
        if W is not None:
            WMSE = ((pred - Y)**2 * W).sum().item()
        else:
            WMSE = np.nan
        if T is not None:
            MSE = ((pred - T)**2).mean().item()
        else:
            MSE = np.nan
        return -WMSE, C_index, MSE

    def prox_loss(self, **kwargs):
        """
        Loss for penalty terms
        """
        lam = kwargs.get("lam", None) # regularization tuning parameter for linear part
        mu = kwargs.get("mu", lam)    # regularization tuning parameter for nonparametric part
        eta = kwargs.get("eta", torch.ones(self.r))   # adaptive weights for linear part 
        zeta = kwargs.get("zeta", torch.ones(self.p)) # adaptive weights for nonparametric part 

        lam = lam * eta
        mu = mu * zeta 
        
        # prox for linear part
        loss1 = (self.part1_layer.weight.abs() * lam).sum()

        # prox for nonlinear part
        W = self.part2_layers[0].weight
        loss2 = (mu*torch.norm(W, p=2, dim=0)).sum()
        
        return loss1+loss2

    def gen_part1_coef(self, Y, W, **tra_X_dic):
        """
        Given the nonparametric part, estimate the linear part (linear coefficients)

        @param Y: observerd survival time
        @param W: weight
        @param lam: regularization tuning parameter
        @param tra_X_dic: training data dictionary (same as forward pass)
        """
        _, _, y2_pred = self.forward(**tra_X_dic)
        X = tra_X_dic[self.X_name]  
        n = X.shape[0]  
        r = (self.part1_important!=0).sum()
        if r > n-1:
            return  
        ones_vector = torch.ones((n, 1)).to(torch.float32)
        X1 = torch.cat((ones_vector, X[:, self.part1_important!=0]), 
                       dim=1)
        regularize = torch.ones(r+1) * n
        regularize[0] = 0
        all_coefs = (torch.inverse((X1.T * W) @ X1)) @ (X1.T * W) @ (Y - y2_pred)  
        self.part1_layer.bias.data[0] = all_coefs[0]  
        self.part1_layer.weight.data[0, self.part1_important!=0] = all_coefs[1:]  


    def gen_adaptive_weights(self, gamma=2):
        """
        Generate adaptive weights based on the estimated model
        @param gamma: tuning parameter for adaptive weights
        @return: eta, adaptive weights for linear part
        @return: zeta, adaptive weights for nonparametric part
        """
        norm_ = self.part1_layer.weight.data.abs().squeeze()  
        if norm_.sum() == 0:
            eta = norm_ + 1000
        else:
            eta = 1/(1e-3 + norm_**gamma)  
            eta = eta / eta.min() * 0.02
        norm_ = ((self.part2_layers[0].weight.data)**2).sum(dim=0).sqrt().float().squeeze()
        if norm_.sum() == 0:
            zeta = norm_ + 1000
        else:
            zeta = 1/(1e-3 + norm_**gamma)  
            zeta = zeta / zeta.min() * 0.005
        return eta, zeta  
    
    def determine_importance(self, thresh1=None, thresh2=None):  
        """
        Given the threshold, determine the important variables for linear and nonparametric parts

        @param thresh1: threshold for linear part
        @param thresh2: threshold for nonparametric part
        """
        if thresh1 is None:
            tmp = (self.part1_layer.weight).abs()
            thresh1 = tmp.std()*1.5 + tmp.mean()  
        if thresh2 is None:
            tmp = ((self.part2_layers[0].weight)**2).sum(dim=0).sqrt()
            thresh2 = tmp.std()*1.5 + tmp.mean()  

        self.part1_important.data = (self.part1_layer.weight.data.abs() > thresh1).float().squeeze() 
        self.part2_important.data = (((self.part2_layers[0].weight.data)**2).sum(dim=0).sqrt() > thresh2).float().squeeze() 
        

    def refit_prox(self):
        """
        Refit prox to zero-out the unimportant variables when conducting refitting
        """
        self.part1_layer.weight.data = self.part1_layer.weight.data * self.part1_important  
        self.part2_layers[0].weight.data = self.part2_layers[0].weight.data * self.part2_important

    def l2_regularization(self, gamma1=0.01, gamma2=0.01, lr=1, **kwargs):
        """
        L2 regularization
        @param gamma1: L2 regularization tuning parameter for linear part
        @param gamma2: L2 regularization tuning parameter for nonparametric part
        """
        reg1 = (torch.norm(self.part1_layer.weight,p=2)** 2)
        reg2 = 0
        for layer in self.part2_layers:
            reg2 += (torch.norm(layer.weight,p=2)** 2)
        return reg1*gamma1 + reg2*gamma2

    def evaluation(self, tes_X_dic=None, tes_Y_dic=None, coef_true=None, coef_g_true=None, **kwargs):
        """
        Evaluation for the estimated model

        @param tes_X_dic: test data dictionary
        @param tes_Y_dic: test data dictionary
        @param coef_true: true linear coefficients
        @param coef_g_true: true nonparametric coefficients
        """
        res = {}
        # Evaluation for overall prediction performance  
        if tes_X_dic is not None and tes_Y_dic is not None:
            self.eval()
            out = self(**tes_X_dic)
            _, _, y2_pred = out
            res["loss"] = self.loss_func(out, **tes_Y_dic).item()
            metrics = self.metric_func(out, **tes_Y_dic)  
            res["metric"] = metrics[0]
            res["metric1"] = metrics[1]
            res["metric2"] = metrics[2]
        else:
            y2_pred = None 

        # Evaluation for variable selection and estimation performance for linear part
        if coef_true is not None:  
            coef_est = self.part1_layer.weight  
            res_coef = eval_linear_coef(coef_true, coef_est)  
            res.update(res_coef)  
        
        # Evaluation for variable selection and estimation performance for nonparametric part
        if coef_g_true is not None:
            res_nonlinear = eval_nonlinear(self.part2_important,  
                                           coef_g_true, y2_pred, tes_Y_dic["y2"]  
                                           )
            res.update(res_nonlinear)  
        return res

    def evaluation_AUC(self, tra_Y_dic, test_Y_dic, tes_X_dic, time_points=None, **kwargs):
        res = {}
        # predict risk scores for test data
        self.eval()
        with torch.no_grad():
            out = self(**tes_X_dic)
        risk_scores = -out[0].detach().numpy()
        # information of test data
        res = evaluation_AUC(tra_Y_dic, test_Y_dic, risk_scores,      
                             time_points=time_points, **kwargs)
        return res  

    def num_nonzero(self):
        """
        Number of non-zero coefficients for linear and nonparametric parts
        """
        p1 = (self.part1_layer.weight.data != 0).sum()
        p2 = (((self.part2_layers[0].weight.data)**2).sum(dim=0) > 1e-5).sum()  
        return p1, p2  
