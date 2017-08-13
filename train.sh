#!/bin/bash
#SBATCH -J spc
#SBATCH -p normal_q
#SBATCH -N 1
#SBATCH -t 20:00:00
export PYTHONUSERBASE=$HOME/hu/python
module load cuda 
source /opt/DL/tensorflow/bin/tensorflow-activate

echo "hello world"
python spc.py

exit;
