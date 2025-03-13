import numpy as np
from scipy.optimize import root
import scipy.stats as stats
import torch

def KM_weights(Y, N_):
    """
    Compute Kaplan-Meier weights

    @param Y: observed time
    @param N_: event indicator
    @return: Kaplan-Meier weights
    """
    n = Y.shape[0]  
    order = Y.argsort()  
    W = np.zeros(n)
    tmp = 1  
    for i in range(n):
        idx = order[i]  
        W[idx] = tmp * N_[idx] / (n-i)  
        tmp *= ((n-i-1)/(n-i))**N_[idx]  
    W = W / W.sum()
    return W

def generate_data(n=1000, n_val=200, n_tes=2000, r=100, p=100,
                  case=0, s1 = 10, 
                  censor_rate=0.3,
                  coef=None, rho=0.5, seed=42
                  ):
        """
        Generate data for simulation

        @param n: number of training samples
        @param n_val: number of validation samples
        @param n_tes: number of test samples
        @param r: number of covariates for X 
        @param p: number of covariates for Z
        @param case: simulation case (simulation examples)
        @param s1: number of non-zero coefficients in X
        @param censor_rate: censor rate
        @param coef: coefficients for X
        @param rho: correlation coefficient
        @param seed: random seed

        @return: training, validation, and test data
        """
        # AFT model
        # ==== initialize ====
        n_all = n+n_val+n_tes

        if not coef:
            np.random.seed(402)
            idxs_sig = np.array(range(s1))
            #idxs_sig = np.random.choice(range(50), size=s1, replace=False)
            coef = np.zeros(r)
            coef[idxs_sig] = 0.8 #np.random.uniform(0.5, 0.7, size=s1) 

        # ==== generate covariates ====
        np.random.seed(seed)
        # generate X
        mean_vec = np.array([0]*r)
        cov_matrix = np.array([[rho**(abs(i-j)) for j in range(r)] for i in range(r)])
        X_all = np.random.multivariate_normal(mean_vec, cov_matrix, size=n_all)  
        #X_all = stats.norm.cdf(X_all)
        X_all[X_all>3] = 3
        X_all[X_all<-3] = -3  
        X_all = X_all/6 + 0.5

        # generate Z
        mean_vec = np.array([0]*p)
        cov_matrix = np.array([[rho**(abs(i-j)) for j in range(p)] for i in range(p)])
        Z_all = np.random.multivariate_normal(mean_vec, cov_matrix, size=n_all)  
        # consider correlation between X and Z
        num_cor = 30
        pre_idx = 20
        H_ = np.array([[0.5**(abs(i-j)) for j in range(num_cor)] for i in range(num_cor)])   
        Z_all_part = X_all[:, pre_idx:(pre_idx+num_cor)].dot(H_) + 0.05*Z_all[:, pre_idx:(pre_idx+num_cor)]  
        mean_ = Z_all_part.mean(axis=0)  
        std_ = Z_all_part.std(axis=0)  
        Z_all_part = (Z_all_part-mean_)/std_  
        Z_all[:, pre_idx:(pre_idx+num_cor)] = Z_all_part
        #
        #Z_all = stats.norm.cdf(Z_all)
        Z_all = (Z_all-Z_all.mean(axis=0)) / Z_all.std(axis=0)
        Z_all[Z_all>3] = 3
        Z_all[Z_all<-3] = -3  
        Z_all = Z_all/6 + 0.5

        ### Calculate survival 
        y1_all = X_all.dot(coef) # linear component
        coef_g = np.zeros(p)  
        if case == 0:   # linear  
            s2 = 6
            idxs_sig = np.array(range(s2))  
            coef_g[idxs_sig] = 1 #np.random.uniform(0.4, 0.6, size=s2)  
            y2_all = Z_all.dot(coef_g) 

        elif case == 1:  # deep 2 (single-index)  
            s2 = 6  
            idxs_sig = np.array(range(s2))  
            coef_g[idxs_sig[:(s2//2)]] = 1  
            coef_g[idxs_sig[(s2//2):s2]] = 1  
            a = Z_all.dot(coef_g)  
            y2_all = 2*a*(5-a)

        elif case == 2:  # additive  
            s2 = 6
            idxs_sig = np.array(range(s2))
            #idxs_sig = np.random.choice(range(p), size=s2, replace=False)
            coef_g[idxs_sig] = 1
            y2_all = (np.sin(2*np.pi*Z_all[:,0]) + \
                    np.sin(2*np.pi*Z_all[:,1]) - \
                    np.cos(2*np.pi*Z_all[:,2]) - \
                    np.cos(2*np.pi*Z_all[:,3]) + \
                    np.sin(2*np.pi*Z_all[:,4]) + \
                    np.sin(2*np.pi*Z_all[:,5])
                    )*1.5
            
        elif case == 3:  # deep 
            s2 = 6
            idxs_sig = np.array(range(s2))
            coef_g[idxs_sig] = 1
            y2_all = (4*Z_all[:,0] * (Z_all[:,1]-0.5)**2 + \
                      np.sqrt(Z_all[:,2]*Z_all[:,3]) + \
                      2*np.sin(2*np.pi*Z_all[:,4]*Z_all[:,5]) )
            
            y2_all = (np.sin(2*np.pi*(Z_all[:,0]**2)*(Z_all[:,1]**2))+\
                      np.log(Z_all[:,2]*Z_all[:,3]+0.5)+\
                      np.abs(np.log(Z_all[:,3]*Z_all[:,4]+0.5))+\
                      np.exp((Z_all[:,4]*Z_all[:,5]))-4  
                      )**3 /4  

        # generate T, C, Y            
        error = np.random.normal(0, 0.8, size=n_all)  
        T_all = y1_all + y2_all + error  # log T  
        T_all = T_all - T_all.mean()
        T_original = np.exp(T_all)  
        f = lambda c_rate: 1-np.exp(-1/c_rate*T_original).mean() - censor_rate  
        c_rate = root(f, np.exp(-2)).x  
        C_all = np.log(np.random.exponential(c_rate, size=n_all))  
        Y_all = np.min(np.stack([T_all, C_all], axis=1), axis=1)  
        N_all = (T_all<=C_all).astype(int)

        # training, validation, and test data
        tra_data = dict(Z=Z_all[:n, :],
                        X=X_all[:n, :],
                        Y=Y_all[:n],
                        N_=N_all[:n],
                        W = KM_weights(Y_all[:n], N_all[:n]), # KM weight for tra data
                        y1=y1_all[:n],
                        y2=y2_all[:n]
                        )
          
        val_data = dict(Z=Z_all[n:(n+n_val), :],
                        X=X_all[n:(n+n_val), :],
                        Y=Y_all[n:(n+n_val)],
                        N_=N_all[n:(n+n_val)],
                        W = KM_weights(Y_all[n:(n+n_val)], N_all[n:(n+n_val)]), 
                        y1=y1_all[n:(n+n_val)],
                        y2=y2_all[n:(n+n_val)]
                        )
        
        tes_data = dict(Z=Z_all[(n + n_val):(n + n_val + n_tes), :],
                        X=X_all[(n + n_val):(n + n_val + n_tes), :],
                        Y=Y_all[(n + n_val):(n + n_val + n_tes)],
                        T=T_all[(n + n_val):(n + n_val + n_tes)],
                        W = KM_weights(Y_all[(n + n_val):(n + n_val + n_tes)], N_all[(n + n_val):(n + n_val + n_tes)]), 
                        N_=N_all[(n + n_val):(n + n_val + n_tes)],
                        y1=y1_all[(n + n_val):(n + n_val + n_tes)],
                        y2=y2_all[(n + n_val):(n + n_val + n_tes)]
                        )
        return tra_data, val_data, tes_data, coef, coef_g


def data_to_torch(data):
    """
    Convert data to torch
    @param data: data
    @return: torch data
    """
    # X
    data_X = {}
    data_X["Z"] = torch.from_numpy(data["Z"]).float()
    data_X["X"] = torch.from_numpy(data["X"]).float()
    # Y
    data_Y = {}
    data_Y["Y"] = torch.from_numpy(data["Y"]).float()
    data_Y["N_"] = torch.from_numpy(data["N_"]).int()
    if "W" in data.keys():
        data_Y["W"] = torch.from_numpy(data["W"]).float()
    if "y1" in data.keys():
         data_Y["y1"] = torch.from_numpy(data["y1"]).float()
    if "y2" in data.keys():
         data_Y["y2"] = torch.from_numpy(data["y1"]).float()
    if "T" in data.keys():
         data_Y["T"] = torch.from_numpy(data["T"]).float()
    return data_X, data_Y


