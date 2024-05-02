#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <stdio.h>
#include <curand_kernel.h>
#include "config.h"
#include "monte_carlo.h"

__global__ void initCurand(unsigned int seed, curandState* states);

__global__ void MC_Heston(curandState* state,
                          float S0,
                          float V0,
                          float r,
                          float kappa,
                          float theta,
                          float rho,
                          float sigma,
                          float dt,
                          float K,
                          int N,
                          float *sum,
                          int n);
                          
#endif
