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

__constant__ static const char seed[32] = {
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

#include <stdint.h>

__global__ void machine_ids(uint32_t *out_buf)
{
  // all threadIdx will be replaced with ced to SR_MACHINE_ID_X
  uint32_t x = threadIdx.x;
  out_buf[0] = x;
  x = threadIdx.y;
  out_buf[1] = x;
  x = threadIdx.z;
  out_buf[2] = x;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(x));
  out_buf[3] = x;
  out_buf[4] = 0x15;
}

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
// #include <cuda_helper.h>
#define checkCudaErrors(err) { \
if (err != cudaSuccess) { \
 fprintf(stderr, "checkCudaErrors() API error = %04d \"%s\" from file <%s>, line %i.\n", \
 err, cudaGetErrorString(err), __FILE__, __LINE__); \
 exit(-1); \
} \
}

// main
__host__ int main()
{
  std::string s; // = "abcdefghijklmnoprstuvwxyz0123456";
  std::cin >> s;
  if ( s.size() != 32 ) {
    printf("bad len of string\n");
    return 1;
  }
  uint32_t *card_id;
  // read card id - 4 * 4 = 16 bytes + 4 for test
  auto err = cudaMalloc(&card_id, 20);
  checkCudaErrors(err);
  machine_ids<<<1,1>>>(card_id);
  err = cudaDeviceSynchronize(); checkCudaErrors(err);
  uint32_t host_card_id[5];
  err = cudaMemcpy(host_card_id, card_id, sizeof(host_card_id), cudaMemcpyDeviceToHost); checkCudaErrors(err);
  // dump card id
  unsigned char *cid = (unsigned char *)host_card_id;
  for ( int i = 0; i < 20; i++ ) printf("%2.2X ", cid[i]);
  fputc('\n', stdout);
  cudaFree(card_id);
  // rest
  char *d_c;
  int *d_i;
  err = cudaMalloc(&d_c, 32); checkCudaErrors(err);
  err = cudaMalloc(&d_i, sizeof(int)); checkCudaErrors(err);
  err = cudaMemcpy(d_c, s.c_str(), 32, cudaMemcpyHostToDevice); checkCudaErrors(err);
  calc_hash<<<1,32>>>(d_c, d_i);
  err = cudaDeviceSynchronize(); checkCudaErrors(err);
  int res = 1;
  err = cudaMemcpy(&res, d_i, sizeof(res), cudaMemcpyDeviceToHost); checkCudaErrors(err);
  cudaFree(d_c);
  cudaFree(d_i);
  if ( res )
   printf("yes\n");
  else
   printf("no\n");
  return res;
}