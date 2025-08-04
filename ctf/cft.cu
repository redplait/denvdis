#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE (1<<5)
// from https://github.com/abdimoallim/cuda-utils/blob/main/cutils.cuh
__device__ __forceinline__ int warp_reduce_sum(int val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

__device__ __forceinline__ int warp_reduce_min(int val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  }
  return val;
}

__device__ static const char seed[32] = {
 0x2f ^ 'a',
 0x2f ^ 'b',
 0x2f ^ 'c',
 0x2f ^ 'd',
 0x2f ^ 'e',
 0x2f ^ 'f',
 0x2f ^ 'g',
 0x2f ^ 'h',
 0x2f ^ 'i',
 0x2f ^ 'j',
 0x2f ^ 'k',
 0x2f ^ 'l',
 0x2f ^ 'm',
 0x2f ^ 'n',
 0x2f ^ 'o',
 0x2f ^ 'p',
 0x2f ^ 'r',
 0x2f ^ 's',
 0x2f ^ 't',
 0x2f ^ 'u',
 0x2f ^ 'v',
 0x2f ^ 'w',
 0x2f ^ 'x',
 0x2f ^ 'y',
 0x2f ^ 'z',
 0x2f ^ '0',
 0x2f ^ '1',
 0x2f ^ '2',
 0x2f ^ '3',
 0x2f ^ '4',
 0x2f ^ '5',
 0x2f ^ '6',
};

__global__ void calc_hash(const char *s, int *res)
{
  int x = threadIdx.x;
  int v = (s[x] ^ seed[x] ^ 0x2f) ? 0 : 1;
  __syncthreads();
  int r = warp_reduce_min(v);
  if ( !x ) {
// printf("res %X\n", r);
    *res = r;
  }
}

#include <string>
#include <iostream>

// main
__host__ int main()
{
  std::string s; // = "abcdefghijklmnoprstuvwxyz0123456";
  std::cin >> s;
  if ( s.size() != 32 ) {
    printf("bad len of string\n");
    return 1;
  }
  char *d_c;
  int *d_i;
  cudaMalloc(&d_c, 32);
  cudaMalloc(&d_i, sizeof(int));
  cudaMemcpy(d_c, s.c_str(), 32, cudaMemcpyHostToDevice);
  calc_hash<<<1,32>>>(d_c, d_i);
  int res = 1;
  cudaMemcpy(&res, d_i, sizeof(res), cudaMemcpyDeviceToHost);
  cudaFree(d_c);
  cudaFree(d_i);
  if ( res )
   printf("yes\n");
  else
   printf("no\n");
  return res;
}