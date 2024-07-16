

echo Things Is Going to happen
source activate base
conda activate flowerTutorial
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    exit 1
fi
nClients=$1
python server.py "$nClients" 
echo Done
