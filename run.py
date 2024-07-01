import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './client_utils'))

sys.path.append(os.path.join(os.path.dirname(__file__), './output'))
import json
from client_utils.ModelEnum import ModelEnum
import pandas as pd
models=['ELASTICNETCV',  'LASSO','MLP_REGRESSOR']

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
    #for DataSet in os.listdir(root_dir):
        
    for data in os.listdir(root_dir):
        if os.path.exists("./output/TimeSeriesFeatures.json"):
            print("removing TimeSeriesFeatures")
            os.remove("./output/TimeSeriesFeatures.json")
        if os.path.exists("./output/SelectedTimeSeriesFeatures.json"):
            print("removing SelectedTimeSeriesFeatures")
            os.remove("./output/SelectedTimeSeriesFeatures.json")
        if os.path.exists("./output/train_data.csv"):
            print("removing train_data")
            os.remove("./output/train_data.csv")
        if os.path.exists("./output/test_data.csv"):
            print("removing test_data")
            os.remove("./output/test_data.csv")
        dataset=os.path.join(root_dir, data)
        print("Dataset: ", dataset)
        for model in models:
            
            cModel = ModelEnum.get_model_data(model)
            model_name = model
            hyperparameters = HP_Generator(cModel[1])
            print("Model: ", model_name)
            for hyperparameter in hyperparameters:
                #########
                #########``
                print("Data", data)
                print("Model: ", model_name)
                print("Hyperparameters: ", hyperparameter)
                print("Hyperparameters: ", hyperparameter)
                toWrite={"model_name":model_name, "HP":hyperparameter}
                with open('./output/hyperParameters.json', 'w') as f:
                    json.dump(toWrite, f)
                sys.stdout.flush()
                os.system('run.bat '+nCleints+' '+dataset)
                sys.stdout.flush()
                
                    
                    
                    
                
            
            