#include <cuda_runtime.h>
#include <iostream>

/*

__global__ tells the compiler this will run on the device

Each thread has a:
- Thread ID (threadIdx.x)
- Block ID (blockIdx.x)
- Block dimension (blockDim.x)

Since we have several blocks, each block has threads from 0 to T so
block 1 will have thread 0 and so will block 2. Since the numbers
repeat, we need to use the combination of 3 dimensions to uniquely
identify a thread.

*/
__global__ void hello_world()
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Hello, World! Thread %d\n", tid);
}

int main()
{
  // We will have 1 block with 10 threads
  hello_world<<<1, 10>>>();

  // Wait until the kernel is done processing
  cudaDeviceSynchronize();

  return 0;
}