#ifndef SAMPLERS_H
#define SAMPLERS_H

#include <stdio.h>
#include <curand_kernel.h>

// Declaration of __device__ functions from samplers.cu

__device__ float GS_star(curandState* state, float alpha);

__device__ float GKM1(curandState* state, float alpha);

__device__ float GKM2(curandState* state, float alpha);

__device__ float GKM3(curandState* state, float alpha);

__global__ void generateSamples(curandState *state, float alpha, float *samples);

#endif
