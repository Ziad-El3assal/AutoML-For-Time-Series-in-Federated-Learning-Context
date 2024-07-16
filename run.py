import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './client_utils'))
import psutil
sys.path.append(os.path.join(os.path.dirname(__file__), './output'))
import json
from client_utils.ModelEnum import ModelEnum
import pandas as pd
import numpy as np
import glob
import time
import subprocess 
models=['XGBOOST_REGRESSOR']

def HP_Generator(Parameters_DICT):
    """
    This Function generates all possible combinations of hyperparameters for a given model
    Parameters_DICT: Dictionary of hyperparameters for the model
    
    yields: A dictionary for a single combination of hyperparameter
    """
    n=len(Parameters_DICT.keys())

    def Recurssive_HP_Generator(Parameters_DICT, n, i, HP):
        if i==n: 
            yield HP
        else:
            for value in Parameters_DICT[list(Parameters_DICT.keys())[i]]:
                HP[list(Parameters_DICT.keys())[i]] = value
                yield from Recurssive_HP_Generator(Parameters_DICT, n, i+1, HP.copy())
    HP = {}
    yield from Recurssive_HP_Generator(Parameters_DICT, n, 0, HP)            
        
        
if __name__ =='__main__' :
    root_dir=sys.argv[1]
    nCleints=sys.argv[2]
    n=1
    pid = os.getpid()

    for data in os.listdir(root_dir):
        if data != '121.csv' and n==1:
            continue
        n+=1
        for file in glob.glob("./output/TimeSeriesFeatures*"):
            print(f"removing {file}")
            os.remove(file)
        
        # Remove files starting with "SelectedTimeSeriesFeatures"
        for file in glob.glob("./output/SelectedTimeSeriesFeatures*"):
            print(f"removing {file}")
            os.remove(file)
        
        # Remove files starting with "train_data"
        for file in glob.glob("./output/train_data*"):
            print(f"removing {file}")
            os.remove(file)
        
        # Remove files starting with "test_data"
        for file in glob.glob("./output/test_data*"):
            print(f"removing {file}")
            os.remove(file)
        
        dataset=os.path.join(root_dir, data)
        print("Dataset: ", dataset)
        for model in models:
            
            cModel = ModelEnum.get_model_data(model)
            model_name = model
            hyperparameters = HP_Generator(cModel[1])
            print("Model: ", model_name)
            for hyperparameter in hyperparameters:
                #########
                #########
                print("Data", data)
                print("Model: ", model_name)
                print("Hyperparameters: ", hyperparameter)
                toWrite={"model_name":model_name, "HP":hyperparameter}
                with open('./output/hyperParameters.json', 'w') as f:
                    json.dump(toWrite, f)
                #run in flowerTutorial environment
                command = ["./tst.sh" ,str(nCleints), dataset]
                subprocess.run(command,timeout=100)
                #command to kill the process
                time.sleep(200)

                # this process PID
                #kill all the python processes except this one
                for proc in psutil.process_iter(['pid', 'name']):
                    try:
                        # Check if the process name is 'python' or 'python3' and if it's not the current process
                        if proc.info['name'] in ('python', 'python3') and proc.info['pid'] != pid:
                            print(f"Killing process {proc.info['pid']} ({proc.info['name']})")
                            proc.kill()
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        pass
                time.sleep(5)

                
                
                    
                    
                    
                
            
            