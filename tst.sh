#!/bin/bash

echo MAGIC Is Going to happen
source activate base
conda activate flowerTutorial

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

export nClients=$1
export DataSetPath=$2

chmod +x dataSetSpliter.py

gnome-terminal -- bash -c "python dataSetSpliter.py "$nClients" "$DataSetPath"" 
echo Data Splited

chmod +x server.py
gnome-terminal -- bash -c "python server.py $nClients" & $serverPID=$!
echo $serverPIDps -eo s,pid

echo Server Started
sleep 5
chmod +x client.py

for ((i=1; i<=$nClients; i++))
do
    echo "Starting Client $i" wiht data Data/split_$i.csv
    sleep 1
    gnome-terminal -- bash -c "python client.py $i ./Data/split_$i.csv" & clientPIDs[$i]=$!
    echo ${clientPIDs[$i]}
done

echo Clients Started

wait $serverPID