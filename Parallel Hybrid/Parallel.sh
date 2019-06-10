#!/bin/bash

## Specifies the interpreting shell for the job.
#$ -S /bin/bash

## Specifies that all environment variables active within the qsub utility be exported to the context of the job.
#$ -V

## Execute the job from the current working directory.
#$ -cwd

## Parallel programming environment (mpich) to instantiate and number of computing slots.
#$ -pe mpich-smp 4
#$ -v  OMP_NUM_THREADS=4
#$ 
## The  name  of  the  job.
#$ -N MPI-03-02

#$ -j y
#$ -o ./3x3-img03-4.out

mpirun -np 1 ./convolution /share/apps/files/convolution/images/im03.ppm /share/apps/files/convolution/kernel/kernel3x3_Edge.txt /state/partition1/im03.ppm 1

