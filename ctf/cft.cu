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

#define XSED 0x2f

__constant__ static const char seed[32] = {
 XSED ^ 'a',
 XSED ^ 'b',
 XSED ^ 'c',
 XSED ^ 'd',
 XSED ^ 'e',
 XSED ^ 'f',
 XSED ^ 'g',
 XSED ^ 'h',
 XSED ^ 'i',
 XSED ^ 'j',
 XSED ^ 'k',
 XSED ^ 'l',
 XSED ^ 'm',
 XSED ^ 'n',
 XSED ^ 'o',
 XSED ^ 'p',
 XSED ^ 'r',
 XSED ^ 's',
 XSED ^ 't',
 XSED ^ 'u',
 XSED ^ 'v',
 XSED ^ 'w',
 XSED ^ 'x',
 XSED ^ 'y',
 XSED ^ 'z',
 XSED ^ '0',
 XSED ^ '1',
 XSED ^ '2',
 XSED ^ '3',
 XSED ^ '4',
 XSED ^ '5',
 XSED ^ '6',
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
  int v = (s[x] ^ seed[x] ^ XSED) ? 0 : 1;
  __syncthreads();
  int r = warp_reduce_min(v);
  if ( !x ) {
// printf("res %X\n", r);
    *res = r;
  }
}

using func_t = void (*)(const char *s, int *res);

__device__ func_t cf1 = calc_hash;

__global__ void dirty_hack(char **what, char **out_res)
{
 printf("cf1 %p what %p value %p calc_hash %p\n", cf1, what, *what, calc_hash);
  *out_res = (char *)&calc_hash;
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

// I found couple of recepits:
// https://leimao.github.io/blog/Pass-Function-Pointers-to-Kernels-CUDA/
// https://forums.developer.nvidia.com/t/array-of-function-pointers-assignment/208952
// both dont work - value of cf1 is nil
template <typename T>
char *try_get_addr(const char *name) {
  T res = nullptr;
  auto err = cudaMemcpyFromSymbol(&res, cf1, sizeof(T));
  if ( !err ) {
    printf("%s: %p\n", name, res);
    if ( res ) return (char *)res;
  }
  if ( err )
    fprintf(stderr, "cudaMemcpyFromSymbol(%s) failed, error = %d (%s)\n", name, err, cudaGetErrorString(err));
  void *f_addr = nullptr;
  err = cudaGetSymbolAddress(&f_addr, cf1);
  if ( err ) {
    fprintf(stderr, "cudaGetSymbolAddress(%s) failed, error = %d (%s)\n", name, err, cudaGetErrorString(err));
    return nullptr;
  }
  // ok, I am stubborn
  err = cudaMemcpy(&res, f_addr, sizeof(T), cudaMemcpyDeviceToHost); checkCudaErrors(err);
  printf("f_addr: %p %p\n", f_addr, res);
  if ( !res ) {
    char *f2;
    err = cudaMalloc(&f2, sizeof(T)); checkCudaErrors(err);
    dirty_hack<<<1,1>>>((char **)f_addr, (char **)f2);
    err = cudaDeviceSynchronize(); checkCudaErrors(err);
    err = cudaMemcpy(&res, f2, sizeof(res), cudaMemcpyDeviceToHost); checkCudaErrors(err);
    cudaFree(f2);
    printf("f_addr2: %p %p\n", f_addr, res);
    if ( res ) {
      return (char *)res;
    }
  }
  return (char *)res;
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
  auto err = cudaMalloc(&card_id, 20); checkCudaErrors(err);
  machine_ids<<<1,1>>>(card_id);
  err = cudaDeviceSynchronize(); checkCudaErrors(err);
  uint32_t host_card_id[5];
  err = cudaMemcpy(host_card_id, card_id, sizeof(host_card_id), cudaMemcpyDeviceToHost); checkCudaErrors(err);
  // dump card id
  unsigned char *cid = (unsigned char *)host_card_id;
  for ( int i = 0; i < 20; i++ ) printf("%2.2X ", cid[i]);
  fputc('\n', stdout);
  cudaFree(card_id);
  // play with symbols
  try_get_addr<func_t>("cf1");
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