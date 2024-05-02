#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "samplers.h"
#include "config.h"
#include "exact.h"

#define NUM_SAMPLES 100
#define BLOCK_SIZE 256


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



__device__ float sample_V(curandState* state,
                          float V0,
                          float r,
                          float kappa,
                          float theta,
                          float rho,
                          float sigma,
                          float T) {

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

__device__ float quadrature_V(curandState* state, 
                              int n_steps, 
                              float V0, 
                              float r, 
                              float kappa, 
                              float theta, 
                              float rho, 
                              float sigma, 
                              float T) {

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
                             int n) {

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

    state[idx] = localState;
}

int main() {
    // Define kernel parameters
    float S0 = 1.0f;
    float V0 = 0.1f;
    float r = 0.0f;
    float kappa = 2.0f;
    float theta = 0.1f;
    float rho = -0.3f;
    float sigma = 0.2f;
    float dt = 0.01f;
    float K = 1.0f;
    int N = 100;
    int n = 1;
    float T = 5.0f;

    curandState *devStates;
    cudaMalloc((void**)&devStates, NUM_SAMPLES * sizeof(curandState));

    float *devSum;
    cudaMalloc((void**)&devSum, NUM_SAMPLES * sizeof(float));

    exact_Heston<<<(NUM_SAMPLES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(devStates,
                                                                              S0, 
                                                                              V0, 
                                                                              r,
                                                                              kappa, 
                                                                              theta,
                                                                              rho, 
                                                                              sigma, 
                                                                              dt, 
                                                                              K, 
                                                                              T, 
                                                                              N, 
                                                                              devSum, 
                                                                              n);

    std::vector<float> hostSum(NUM_SAMPLES);
    cudaMemcpy(hostSum.data(), devSum, NUM_SAMPLES * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_SAMPLES; ++i) {
        std::cout << "Sample " << i << ": " << hostSum[i] << std::endl;
    }

    cudaFree(devStates);
    cudaFree(devSum);

    return 0;
}
