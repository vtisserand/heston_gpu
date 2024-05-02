#include <stdio.h>
#include <curand_kernel.h>
#include "config.h"
#include "monte_carlo.h"

__global__ void initCurand(unsigned int seed, 
						   curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

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
                          int n) {

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localState = state[idx];
    float2 Ws = make_float2(0.0f, 0.0f); // Initialize Ws

    float S = S0;
    float V = V0;

    // Dynamic allocation of shared memory
    extern __shared__ float A[];
    float* R1s = A;
    float* R2s = R1s + blockDim.x;

    for (int i = 0; i < N; i++) {
        Ws = curand_normal2(&localState);
        V += kappa * (theta - V) * dt * dt + sigma * sqrtf(fmaxf(0.0f, V)) * dt * Ws.x;
        S += r * S * dt * dt + sqrtf(V) * S * dt * (rho * Ws.x + sqrtf(1 - rho * rho) * Ws.y);
    }

    R1s[threadIdx.x] = expf(-r * dt * dt * N) * fmaxf(0.0f, S - K); // Call price
    R2s[threadIdx.x] = R1s[threadIdx.x] * R1s[threadIdx.x];

    __syncthreads(); // Block-level synchronisation

    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            R1s[threadIdx.x] += R1s[threadIdx.x + i];
            R2s[threadIdx.x] += R2s[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Update global memory atomically
    if (threadIdx.x == 0) {
        atomicAdd(sum, R1s[0] / n);
        atomicAdd(sum + 1, R2s[0] / n);
    }
}


int main(void) {

	int NTPB = 1024;
	int NB = 1024;
	int n = NB * NTPB;
	float T = 5.0f;
	float S0 = 1.0f;
	float V0 = 0.1f;
	float K = 1.0;
	float sigma = 0.2f;
	float r = 0.0f;
    float kappa = 2.0f;
    float theta = 0.1f;
    float rho = -0.3f;
	int N = 1000;
	float dt = sqrtf(T/N);

	// Allow some memory for sum (options payoff at maturity) and random states.
	float *sum;
	cudaMallocManaged(&sum, 2*sizeof(float)); // We save payoff and std
	cudaMemset(sum, 0, 2*sizeof(float));

	curandState* states;
	cudaMalloc(&states, n*sizeof(curandState));
	initCurand<<<NB, NTPB>>>(time(0), states);

	float Tim;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Launch one simulation on each thread.
	MC_Heston<<<NB, NTPB, 2*NTPB*sizeof(float)>>>(states, 
												  S0, 
												  V0, 
												  r, 
												  kappa, 
												  theta, 
												  rho, 
												  sigma, 
												  dt, 
												  K, 
												  N, 
												  sum, 
											 	  n);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&Tim, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("The estimated price is equal to %f\n", sum[0]);
	printf("error associated to a confidence interval of 95%% = %f\n",
		1.96 * sqrt((double)(sum[1] - (sum[0] * sum[0])))/sqrt((double)n));
	printf("Execution time %f ms\n", Tim);

	cudaFree(sum);
	cudaFree(states);

	return 0;
}