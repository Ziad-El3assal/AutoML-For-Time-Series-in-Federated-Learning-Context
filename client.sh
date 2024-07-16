
source activate base
conda activate flowerTutorial
if [ "$#" -ne 2]; then
    echo "Illegal number of parameters"
    exit 1
fi
echo client $clientNum Initated

clientNum=$1
dataset=$2
python client.py "$clientNum" "$dataset"

