#include <cuda_runtime.h>
#include <iostream>

#define N 10'000'000 // Number of speed measurements
#define T 0.25       // Time interval between measurements (15 mins)

__global__ void trapezoidalKernel(double *speeds, double *result, int n)
{
  extern __shared__ double sharedData[];
  int tid = threadIdx.x;
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  double localdistance = 0.0;
  if (i < n - 1)
    localdistance = 0.5 * (speeds[i] + speeds[i + 1]) * T;

  sharedData[tid] = localdistance;
  __syncthreads();
}