#!/bin/bash
#BSUB -J poisson_nonblock                                                                                      
#BSUB -q hpcintro                                                                                              
#BSUB -W 1:00
#BSUB -M 4GB                                                                                                   
#BSUB -n 24                                                                                                    
#BSUB -R "span[hosts=1]"                                                                                       
#BSUB -o jobs/output/poisson_nonblock_%J.out
#BSUB -e jobs/output/poisson_nonblock_%J.err

mkdir -p results jobs/output

module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11 >& /dev/null
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8 >& /dev/null

lscpu | grep -E "Model name|cache"
                                                                                    
N=200
ITER=100
RANKS="1 2 4 8 12 16 24"
MOPTS="--bind-to core"

CSV_OUT="results/mpi_nonblocking.csv"
CSV_RANKS="results/mpi_nonblocking_ranks.csv"

rm -f $CSV_OUT $CSV_RANKS

for NP in $RANKS; do
    echo "Executant No-Blocking bàsic amb $NP ranks..."
    mpirun -n $NP $MOPTS python3 mpi_nonblocking.py -N $N --max-iter $ITER --tol 0 --csv $CSV_OUT --rank-csv   
    $CSV_RANKS
done
