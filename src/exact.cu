#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "samplers.h"
#include "config.h"
#include "exact.h"


__device__ float sample_V(curandState* state, float V0, float r, float kappa, float theta, float rho, float sigma, float T){
	float delta = (4 * theta * kappa) / (sigma * sigma);
	float lambda = (4 * kappa * expf(-kappa * T)) / (sigma * sigma * (1 - expf(-kappa * T)));

	// Sample from a non-central chi-squared:
    float chi_2;
    float non_central;
	if (delta > 1.0f) {
		// First, chi-squared of degree (\delta - 1), check for alpha values in gamma sample.
		if ((delta - 1) / 2 > 1) {
			chi_2 = 2 * GKM3(state, (delta - 1) / 2);
		}
		else {
			chi_2 = 2 * GS_star(state, (delta - 1) / 2);
		}
		float eps = curand_normal(state);
		float fact = (eps + sqrtf(lambda));
		non_central = fact * fact + chi_2;
	}
	else{
		// We need to sample from a Poisson here
		float poiss = curand_poisson(state, lambda / 2);
		float degree = delta + 2 * poiss;
		if (degree / 2 > 1) {
			chi_2 = 2 * GKM3(state, degree / 2);
		}
		else {
			chi_2 = 2 * GS_star(state, degree / 2);
		}
		non_central = chi_2;
	}
	return ((sigma * sigma * (1 - expf(-kappa * T)))) / (4 * kappa) * non_central;
}

__device__ float quadrature_V(curandState* state, int n_steps, float V0, float r, float kappa, float theta, float rho, float sigma, float T) {
    float dt = T / n_steps;

    float integral = 0.0f;
    float V_prev = V0;

    for (int i = 1; i <= n_steps; ++i) {
        float t = i * dt;
        float V_current = sample_V(state, V_prev, r, kappa, theta, rho, sigma, t);

        // Trapeizodal integration
        integral += 0.5f * (V_prev + V_current) * dt;
        V_prev = V_current;
    }

    return integral;
}


__global__ void exact_Heston(float S0, float V0, float r, float kappa, float theta, float rho, float sigma, float dt, float K, float T,
                    int N, curandState *state, float *sum, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState = state[idx];
    int n_steps = 100;

    // Careful here, currently we are not sampling (V, \int V ds) but rather them independently
    float V = sample_V(&localState, V0, r, kappa, theta, rho, sigma, T);
    float int_V = quadrature_V(&localState, n_steps, V0, r, kappa, theta, rho, sigma, T);
    float stoch_int = 1 / sigma * (V - V0 - kappa * theta * T + kappa * int_V);

    float mean = logf(S0) + r * T - 1 / 2 * int_V + rho * stoch_int;
    float var = (1 - rho * rho) * int_V;

    float rand = curand_normal(&localState);
    sum[idx] = mean + sqrtf(var) * rand;
}

int main() {
    // Define kernel parameters
    float S0 = 100.0f;
    float V0 = 0.09f;
    float r = 0.05f;
    float kappa = 2.0f;
    float theta = 0.09f;
    float rho = -0.3f;
    float sigma = 1.0f;
    float dt = 0.01f;
    float K = 100.0f;
    int N = 100;
    int n = 1;
    float T = 5.0f;

    // Allocate memory for random states on GPU
    curandState *devStates;
    cudaMalloc((void**)&devStates, NUM_SAMPLES * sizeof(curandState));

    // Allocate memory for sum on GPU
    float *devSum;
    cudaMalloc((void**)&devSum, NUM_SAMPLES * sizeof(float));

    // Launch the kernel
    exact_Heston<<<(NUM_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        S0, V0, r, kappa, theta, rho, sigma, dt, K, T, N, devStates, devSum, n);

    // Copy sum from GPU to CPU
    std::vector<float> hostSum(NUM_SAMPLES);
    cudaMemcpy(hostSum.data(), devSum, NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < NUM_SAMPLES; ++i) {
        std::cout << "Sample " << i << ": " << hostSum[i] << std::endl;
    }

    // Free allocated memory on GPU
    cudaFree(devStates);
    cudaFree(devSum);

    return 0;
}
