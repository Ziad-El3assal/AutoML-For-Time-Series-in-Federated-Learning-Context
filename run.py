import sys
import os
import glob
sys.path.append(os.path.join(os.path.dirname(__file__), './client_utils'))

sys.path.append(os.path.join(os.path.dirname(__file__), './output'))
import json
from client_utils.ModelEnum import ModelEnum
import pandas as pd
models=['ELASTICNETCV']

df = pd.read_csv('output\FLresults.csv') 
# print(df.dtypes)

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


def check_hyperparameters(row, hyperparameters_to_check):
    try:
        hyperparameters = eval(row['hyperparameters'])  # Convert string representation to dictionary
        # print("Hyperparameters:", hyperparameters, "Type:", type(hyperparameters))  # Print the hyperparameters and their type
        for key, value in hyperparameters_to_check.items():
            # print(key)
            # print(value)
            if key not in hyperparameters or hyperparameters[key] != value:
                return False
        return True
    except Exception as e:
        print(f"Error evaluating hyperparameters: {e}")
        return False
        
        
if __name__ =='__main__' :
    # print(df.dtypes)
    root_dir=sys.argv[1]
    nCleints=sys.argv[2]
    #for DataSet in os.listdir(root_dir):
        
    for data in os.listdir(root_dir):
        # if os.path.exists("./output/TimeSeriesFeatures.json"):
        #     print("removing TimeSeriesFeatures")
        #     os.remove("./output/TimeSeriesFeatures.json")
        # if os.path.exists("./output/SelectedTimeSeriesFeatures.json"):
        #     print("removing SelectedTimeSeriesFeatures")
        #     os.remove("./output/SelectedTimeSeriesFeatures.json")
        # if os.path.exists("./output/train_data.csv"):
        #     print("removing train_data")
        #     os.remove("./output/train_data.csv")
        # if os.path.exists("./output/test_data.csv"):
        #     print("removing test_data")
        #     os.remove("./output/test_data.csv")
        # dataset=os.path.join(root_dir, data)
        # print("Dataset: ", dataset)
       
        # Remove files starting with "TimeSeriesFeatures"
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
        
        dataset = os.path.join(root_dir, data)
        print("Dataset: ", dataset)
        for model in models:
            
            cModel = ModelEnum.get_model_data(model)
            model_name = model
            hyperparameters = HP_Generator(cModel[1])
            print("Model: ", model_name)
            for hyperparameter in hyperparameters:
                #########
                #########`
                found = False
                # Iterate through the DataFrame and check the conditions
                for index, row in df.iterrows():
                    # print(df.dtypes)
                    # print("model")
                    # print(row['model'].lower() == model_name.lower())
                    # print("dataset_name")
                    # print(str(row['dataset_name']) == os.path.splitext(os.path.basename(dataset))[0])
                    # print(str(row['dataset_name']))
                    # print(str(dataset))
                    # print("num_clients")
                    # print(str(row['num_clients']) == str(nCleints))
                    # print("hyper")
                    # print(check_hyperparameters(row, hyperparameter))
                    if (row['model'].lower() == model_name.lower() and 
                        str(row['dataset_name']) == os.path.splitext(os.path.basename(dataset))[0] and 
                        str(row['num_clients']) == str(nCleints) and 
                        check_hyperparameters(row, hyperparameter)):
                            print(f"record already exist at row number {index} .")
                            found = True
                            break
                if not found:
                    print("No matching row found.")
                    print("Data", data)
                    print("Model: ", model_name)
                    print("Hyperparameters: ", hyperparameter)
                    print("Hyperparameters: ", type(hyperparameter))
                    toWrite={"model_name":model_name, "HP":hyperparameter}
                    with open('./output/hyperParameters.json', 'w') as f:
                        json.dump(toWrite, f)
                    sys.stdout.flush()
                    os.system('run.bat '+nCleints+' '+dataset)
                    sys.stdout.flush()
                
                    
                    
                    
                
            
            