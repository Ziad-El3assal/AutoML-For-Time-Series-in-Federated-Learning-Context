##check records
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), './client_utils'))

sys.path.append(os.path.join(os.path.dirname(__file__), './output'))
import json
from client_utils.ModelEnum import ModelEnum
import pandas as pd
models=['LASSO']

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

if __name__ == '__main__':
    root_dir = sys.argv[1]
    nClients = sys.argv[2]
    processed_datasets = set()

    #load processed datasets list 
    processed_datasets_file = './output/processed_datasets.txt'
    if os.path.exists(processed_datasets_file):
        with open(processed_datasets_file, 'r') as f:
            processed_datasets = set(f.read().strip().split('\n'))     #read ds as set

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
        dataset = os.path.join(root_dir, data)
        
        #skip dataset if already processed
        if data in processed_datasets:
            print(f"Skipping dataset {data} as its already processed.")
            continue
        
        #if its not, process it
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
                os.system('run.bat '+nClients+' '+dataset)  #nCleints
                sys.stdout.flush()
                
        #after processing a dataset, record it as processed
        processed_datasets.add(data)
        
        #write updated list of processed datasets to file
        with open(processed_datasets_file, 'w') as f:
            f.write('\n'.join(processed_datasets))

                
            
            