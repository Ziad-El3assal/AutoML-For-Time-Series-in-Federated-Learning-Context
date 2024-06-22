

import pandas as pd 
import sys
import os
import shutil

def partition_data(data_path, nclients):
    """
    Partitions the data into nclients parts.
    """
    data = pd.read_csv(data_path)
    data_length = len(data)
    partition_size = data_length // nclients
    partitions = []
    for i in range(nclients):
        start = i * partition_size
        end = (i + 1) * partition_size
        if i == nclients - 1:
            end = data_length
        partitions.append(data.iloc[start:end])
    return partitions

def create_Parttioned_files(data_path, nclients):
    """
    Partitions the data into nclients parts and saves each part into a separate file.
    """
    partitions = partition_data(data_path, nclients)
    #create a folder for the data
    if os.path.exists("Data"):
        # Remove the directory and all its contents
        shutil.rmtree("Data")
        os.makedirs("Data")
    else:
        os.mkdir("Data")
        
    for i, partition in enumerate(partitions):
        partition.to_csv(f"Data\split_{i + 1}.csv", index=False)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python dataSetSpliter.py <data_path> <nclients>")
        sys.exit(1) 
    data_path = sys.argv[1]
    nclients = int(sys.argv[2])
    create_Parttioned_files(data_path, nclients)
    print(f"Data partitioned into {nclients} parts and saved in the Data folder.")