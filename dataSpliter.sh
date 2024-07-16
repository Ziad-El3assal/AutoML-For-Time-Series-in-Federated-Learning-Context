
#!/bin/bash

echo Things Is Going to happen
source activate base
conda activate flowerTutorial

if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    exit 1
fi

# Set and export variables
export nClients=$1
export DataSetPath=$2

# Print the variables to ensure they are set
echo "nClients: $nClients"
echo "DataSetPath: $DataSetPath"

python dataSetSpliter.py "$nClients" "$DataSetPath"
echo $DataSetPath
echo Done
