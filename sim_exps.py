from sim_funcs_architectures import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--id", default=0) # id of the setting in `sim_schemes`
parser.add_argument("--reps", default=100) # repeat times for each setting

if __name__ == "__main__":  
    args = parser.parse_args()
    # =============================================  
    sim_schemes = pd.read_excel("sim_schemes_final.xlsx")  
    sim_schemes.index = sim_schemes["id"]  
    rep_times = int(args.reps)    # repeat times for each setting  
    folder = f"./result/"   # folder for saving results  
    logger_file = os.path.join(folder, "training_7705.log")  
    # =============================================     
    if not os.path.exists(folder):  
        os.mkdir(folder)  
    logger = creater_logger(logger_file)  
    i = int(args.id)
    logger.info(f"==== setting {i} ====")  
    setting = get_settings(sim_schemes, i)                  # get the setting from `sim_schemes`
    setting["metrics"] = ["metric", "metric1", "metric2",   # output metrics names
                "TP", "FP", "TP_g", "FP_g", "SSE", 
                "TPR", "FPR", "TPR_g", "FPR_g", "Time"]  
    label = str(i)  

    #### run simulation ####
    sim_multi_times(range(100, 100+rep_times), 
                    sim_func, setting, folder_path=folder, label=label
                    )  

    logger.info(f"==== setting {i} END ====")  


