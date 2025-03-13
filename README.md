# Paper

Integrating Genetic and Pathological Imaging Data for Cancer Prognosis with a DNN-based Semiparametric Model (forthcoming)

# Maintainer

Jingmao Li,  [jingmao.li@yale.edu](jingmao.li@yale.edu)  

# Files and functions

* `Model.py`  
  This file contains the `torch.nn` class for the proposed DNN-based Semiparametric AFT model  
  Main class:
  * `Class Model_single_modal`  
    Proposed DNN-based Semiparametric AFT model

* `Trainer.py`  
  This file contains the functions for model training and evaluation  
  Main class:
  * `class Trainer`  
  Class used for model training and evaluation

* `DGP.py`  
  This file contains the function for similation data generation  
  Main function:
  * `generate_data`  
    The function used to generate simulation data

* `sim_funcs_architectures.py`  
  This file contains the function for conducting similation for the proposed method.  
  Main functions:  
  * `sim_func`
    The function used in simulation study 
  * `sim_multi_times`
    The function used to run simulations for multiple times
* `sim_exps.py`  
  This is the main file, which conducts simulations for various settings. The parallel runing is used to fasten the computation.
* `sim_schemes_final.xlsx`  
  The table containing the simulation settings. 
  * case=0: Example 1, linear case
  * case=1: Example 2, single-index case
  * case=2: Example 3, additive case
  * case=3: Example 4, deep composite case

* `utils.py`  
  Some util functions used in the project. 

# Usage

* Use the command line to conduct the simulation under the specific setting (indexed by parameter `--id`) in `sim_schemes_final.xlsx`. Another parameter `--reps` represents the number of replications for the simulation, e.g., 20, 100.  
    Usage example:
    ```bash
    python sim_exps.py --id 0 --reps 100
    ```

