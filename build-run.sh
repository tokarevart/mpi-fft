#!/bin/bash
mpicc main.c -o main -lm -O3 -std=c11 &&
sbatch -n 4 -t 1 -p debug --wrap "mpiexec ./main"