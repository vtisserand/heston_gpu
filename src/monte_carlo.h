#ifndef MONTE_CARLO_H
#define MONTE_CARLO_H

#include <curand_kernel.h>

// Declaration of the CUDA kernel
__global__ void MC_Heston(float S0, float V0, float r, float kappa, float theta, float rho, float sigma, float dt, float K, 
                           int N, curandState *state, float *sum, int n);

#endif
