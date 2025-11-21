#!/bin/bash

### Directivas para el gestor de colas
#SBATCH --job-name=sAXPYp
#SBATCH -D .
#SBATCH --output=out/submit-sAXPYp.o%j
#SBATCH --error=out/submit-sAXPYp.e%j
#SBATCH -A cuda
#SBATCH -p cuda

## SOLO 1 DE LAS TRES OPCIONES PUEDE ESTAR ACTIVA
## OPCION A: Usamos la RTX 4090
##SBATCH --qos=cuda4090  
##SBATCH --gres=gpu:rtx4090:1

## OPCION B: Usamos las 4 RTX 3080
##SBATCH --qos=cuda3080  
##SBATCH --gres=gpu:rtx3080:4

## OPCION C: Usamos 1 RTX 3080
#SBATCH --qos=cuda3080  
#SBATCH --gres=gpu:rtx3080:1

export PATH=/Soft/cuda/12.2.2/bin:$PATH


./build/cuda_simulator





