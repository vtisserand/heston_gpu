#ifndef EXACT_H
#define EXACT_H

#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "samplers.h"

__device__ float sample_V(curandState* state, 
                          float V0, 
                          float r, 
                          float kappa, 
                          float theta, 
                          float rho, 
                          float sigma, 
                          float T);

__device__ float quadrature_V(curandState* state, 
                              int n_steps, 
                              float V0, 
                              float r, 
                              float kappa, 
                              float theta, 
                              float rho, 
                              float sigma, 
                              float T);

__global__ void exact_Heston(curandState *state, 
                             float S0, 
                             float V0, 
                             float r, 
                             float kappa, 
                             float theta, 
                             float rho, 
                             float sigma, 
                             float dt, 
                             float K, 
                             float T,
                             int N, 
                             float *sum, 
                             int n);

#endif
