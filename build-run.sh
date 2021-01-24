#!/bin/bash
mpicc main.c -o main -lm -O3 -std=c11
while true; do
    read -p 'n = ' nproc
    sbatch -n $nproc -t 1 -p debug --wrap "mpiexec ./main"
done