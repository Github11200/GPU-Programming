#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// __global__ void checkPrimeKernel(long long start, long long end, int *primes, bool *is_prime)
// {
//   int tid = threadIdx.x + blockIdx.x * blockDim.x;

//   // Calculate what number we're using based on the thread id
//   long long num = start + (tid * 2);
//   primes[tid] = num;

//   if (num > end)
//     return;
//   bool isPrime = true;
//   if (num <= 1)
//   {
//     is_prime[tid] = false;
//     return;
//   }
//   if (num == 2)
//   {
//     is_prime[tid] = true;
//     return;
//   }

//   if (num % 2 == 0)
//   {
//     is_prime[tid] = false;
//     return;
//   }

//   for (long long i = 3; i * i <= num; i += 2)
//   {
//     if (num % i == 0)
//     {
//       is_prime[tid] = false;
//       break;
//     }
//   }
// }

__global__ void checkPrimeKernel(long long start, long long end, long long *primes, bool *is_prime)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  long long num = start + (tid * 2);
  if (num > end)
    return;

  bool isPrime = true;
  if (num <= 1)
    isPrime = false;
  else if (num % 2 == 0)
    isPrime = (num == 2);

  for (long long i = 3; i * i <= num; i += 2)
  {
    if (num % i == 0)
    {
      isPrime = false;
      break;
    }
  }

  primes[tid] = num;
  is_prime[tid] = isPrime;
}

bool checkPrimeCpu(long long num)
{
  if (num <= 1)
    return false;
  if (num == 2)
    return true;
  if (num % 2 == 0)
    return false;
  for (long long i = 3; i * i <= num; i += 2)
  {
    if (num % i == 0)
    {
      return false;
    }
  }
  return true;
}

/*

CUDA Events are used for measuruing the time elapsed on the GPU
the way chronos is for the CPU.

*/
int main()
{
  long long start = 100'001LL; // must start with odd
  long long end = 190'001LL;

  int threadsPerBlock = 256;
  int totalNumbers = (end - start) / 2 + 1;
  int blocksPerGrid = (totalNumbers + threadsPerBlock - 1) / threadsPerBlock;

  size_t bytesNumbers = totalNumbers * sizeof(long long);
  size_t bytesFlags = totalNumbers * sizeof(bool);

  long long *h_primes = (long long *)malloc(bytesNumbers);
  bool *h_is_prime = (bool *)malloc(bytesFlags);

  long long *d_primes;
  bool *d_is_prime;
  cudaMalloc((void **)&d_primes, bytesNumbers);
  cudaMalloc((void **)&d_is_prime, bytesFlags);

  cudaMemcpy(d_primes, h_primes, bytesNumbers, cudaMemcpyHostToDevice);
  cudaMemcpy(d_is_prime, h_is_prime, bytesFlags, cudaMemcpyHostToDevice);

  cudaEvent_t startEvent,
      stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);

  cudaEventRecord(startEvent, 0);

  checkPrimeKernel<<<blocksPerGrid, threadsPerBlock>>>(start, end, d_primes, d_is_prime);

  cudaEventRecord(stopEvent, 0);
  cudaEventSynchronize(stopEvent); // Blocking call that waits for the device to finish the kernel execuation

  float gpuDuration = 0;
  cudaEventElapsedTime(&gpuDuration, startEvent, stopEvent);

  std::cout << "Time taken on GPU: " << gpuDuration << " ms" << std::endl;

  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);

  cudaMemcpy(h_primes, d_primes, bytesNumbers, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_is_prime, d_is_prime, bytesFlags, cudaMemcpyDeviceToHost);

  for (int i = 0; i < totalNumbers; ++i)
  {
    std::cout << h_primes[i] << ", " << h_is_prime[i] << std::endl;
  }

  auto startTime = std::chrono::high_resolution_clock::now();

  for (long long num = start; num <= end; num += 2)
  {
    checkPrimeCpu(num);
  }

  auto endTime = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> cpuDuration = endTime - startTime;

  std::cout << "Time taken on CPU: " << std::fixed << cpuDuration.count() << " ms" << std::endl;
  std::cout << "speed up : " << cpuDuration.count() / gpuDuration << std::endl;

  return 0;
}