#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "samplers.h"

#define NUM_SAMPLES 1000000
#define BLOCK_SIZE 256

__global__ void initCurand(unsigned int seed, 
                           curandState* states) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}


__device__ float GS_star(curandState* state, 
                         float alpha) {

    float e = 2.718;
	float b = (alpha + e) / e;

	float u, y, z, w, w_1;
    do {
        u = curand_uniform(state);
		y = b*u;
        if (y <= 1.0f) {
            z = powf(y, 1/alpha);
			float temp = curand_uniform(state);
			w = -logf(temp); // Inverse cdf.
			if (w > z)
				return z;
        } else {
            z = -logf((b-y) / alpha);
			w_1 = curand_uniform(state);
			w = powf(w_1, 1 / (alpha - 1));
			if (w <= z)
                return z;
        }
    } while (true);

    return z;
}

__device__ float GKM1(curandState* state, 
                      float alpha) {

    float a = alpha - 1;
	float b = (alpha - 1 / (6*alpha)) / a;
	float m = 2/1;
	float d = m + 2;

	float x_prime, y_prime, v;
    do {
        x_prime = curand_uniform(state);
        y_prime = curand_uniform(state);
		    v = b * y_prime / x_prime;
        if (m * x_prime - d + v + 1/v <= 0.0f) {
            return a * v;
        } else {
            if (m * logf(x_prime) - logf(v) + v - 1 <= 0.0f) {
                return a * v;
            }
        }
    } while (true);
    return 0.0f; // Unreachable
}

__device__ float GKM2(curandState* state, 
                      float alpha) {

    float a = alpha - 1;
	float b = (alpha - 1 / (6*alpha)) / a;
	float m = 2/1;
	float d = m + 2;
	float f = sqrtf(alpha);

	float x, y_prime, x_prime;
    do {
        x = curand_uniform(state);
        y_prime = curand_uniform(state);
		x_prime = y_prime + (1 - 1.857764 * x) / f;
    } while (x_prime > 0.0f and x_prime < 1.0f);

	float v;
	do {
        v = b * y_prime / x_prime;
        if (m * x_prime - d + v + 1/v <= 0.0f) {
            return a * v;
        } else {
            if (m * logf(x_prime) - logf(v) + v - 1 <= 0.0f) {
                return a * v;
            }
        }
    } while (true);

    return 0.0f; // Unreachable
}

__device__ float GKM3(curandState* state,  
                      float alpha) {

    float alpha_0 = 2.5f;
	if (alpha < alpha_0) {
		return GKM1(state, alpha);
	}
	else {
		return GKM2(state, alpha);
	}
    return 0.0f; // Unreachable
}

// Define the CUDA kernel to generate samples
__global__ void generateSamples(curandState *state, 
                                float alpha, 
                                float *samples) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState = state[idx];

    // Generate samples using GS_star function
    samples[idx] = GS_star(&localState, alpha);

    state[idx] = localState;
}


int main() {
    // Allocate memory for random states on GPU
    curandState *devStates;
    cudaMalloc((void**)&devStates, NUM_SAMPLES * sizeof(curandState));

    // Initialize random states on GPU
    initCurand<<<(NUM_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(time(0), devStates);

    // Allocate memory for samples on GPU
    float *devSamples;
    cudaMalloc((void**)&devSamples, NUM_SAMPLES * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    float alpha = 1.5f;
    generateSamples<<<(NUM_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devStates, alpha, devSamples);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken to sample " << NUM_SAMPLES << " samples: " << milliseconds << " ms" << std::endl;

    // Copy samples from GPU to CPU
    std::vector<float> hostSamples(NUM_SAMPLES);
    cudaMemcpy(hostSamples.data(), devSamples, NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);

    // Save samples to CSV file
    std::ofstream outputFile("samples.csv");
    if (outputFile.is_open()) {
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            outputFile << hostSamples[i] << std::endl;
        }
        outputFile.close();
        std::cout << "Samples saved to samples.csv" << std::endl;
    } else {
        std::cerr << "Error: Unable to open samples.csv for writing" << std::endl;
    }

    cudaFree(devStates);
    cudaFree(devSamples);

    return 0;
}