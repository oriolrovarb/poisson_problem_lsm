#!/bin/bash
#BSUB -J poisson_prof
#BSUB -q hpcintro
#BSUB -W 1:00
#BSUB -M 4GB
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "affinity[core(1):distribute=balance]"
#BSUB -o jobs/output/poisson_prof_%J.out
#BSUB -e jobs/output/poisson_prof_%J.err

module load matplotlib/3.10.3-numpy-2.3.1-python-3.12.11 >& /dev/null
module load mpi4py/4.0.3-python-3.12.11-openmpi-5.0.8 >& /dev/null
module load mpiP

lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket"

# Force OpenMPI mapping/binding via environment so mpiPrun cannot ignore it
export OMPI_MCA_rmaps_base_mapping_policy="socket"
export OMPI_MCA_hwloc_base_binding_policy="core"

ITER=100

for N in 200 400; do
    rm -rf results/cProf_N${N}
    mpirun -n 16 --output dir=results/cProf_N${N} \
        python3 -m cProfile -s cumulative main_mpi.py -N $N --max-iter $ITER --tol 0

    mkdir -p results/mpiP_N${N}
    export MPIP="-d -f results/mpiP_N${N}"
    mpiPrun -n 16 python3 main_mpi.py -N $N --max-iter $ITER --tol 0
done