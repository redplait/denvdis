// Programming in Parallel with CUDA - supporting code by Richard Ansorge 
// copyright 2021 is licensed under CC BY-NC 4.0 for non-commercial use
// This code may be freely changed but please retain an acknowledgement

// program gpusum_tla includes:
// example 2.1 tls using while loop
// example 2.2 tla using for   loop
// same code with 5th argument to choose method

#include "cx.h"
#include "cxtimers.h"                    // timers


#define WARP_SIZE (1<<5)

__device__ __forceinline__ float warp_reduce_sum(float val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }

  return val;
}

__host__ __device__ inline float sinsum(float x,int terms)
{
    float x2 = x*x;
    float term = x;   // first term of series
    float sum = term; // sum of terms so far
    for(int n = 1; n < terms; n++){
	term *= -x2 / (2*n*(2*n+1));
	sum += term;
    }
    return sum;
}

// kernel uses many parallel threads for sinsum calls
__global__ void gpu_sin_tla_whileloop(float *sums,int steps,int terms,float step_size)
{
    int step = blockIdx.x*blockDim.x+threadIdx.x; // start with unique thread ID
    int res_idx = step >> 5;
    float res = 0.0;
    auto stride = blockDim.x*gridDim.x;
    while(step<steps){
	float x = step_size*step;
	res += sinsum(x,terms);  // save sum
	step += stride; //  large stride to next step.
    }
    __syncthreads();
    sums[res_idx] = warp_reduce_sum(res);
}

int main(int argc,char *argv[])
{
    if (argc < 2) {
	printf("usage gpusum_tla steps|1000000 trems|1000 threads|256 blcoks|256\n");
	return 0;
    }
    int steps   = (argc > 1) ? atoi(argv[1])  : 1000000;
    int terms   = (argc > 2) ? atoi(argv[2])  : 1000;
    int threads = (argc > 3) ? atoi(argv[3])  : 256;
    // int blocks  = (argc > 4) ? atoi(argv[4])  : (steps+threads-1)/(4 * threads);  // ensure threads*blocks >= steps

    double pi = 3.14159265358979323;
    double step_size = pi / (steps-1); // NB n-1 steps between n points

    auto cl = [=](int scale) {
      int blocks = (steps+threads-1)/(scale * threads);
      int sw = scale * WARP_SIZE;
      int dsize = steps / sw;
      if ( steps & (sw - 1) ) dsize++;
  printf("scale %d dsums size %d\n", scale, dsize);
      thrust::device_vector<float> dsums(dsize);         // GPU buffer    
      float *dptr = thrust::raw_pointer_cast(&dsums[0]); // get pointer    

      cx::timer tim;                  // declare and start timer
      gpu_sin_tla_whileloop<<<blocks,threads>>>(dptr,steps,terms,(float)step_size);  // tla using while loop
      double gpu_sum = thrust::reduce(dsums.begin(),dsums.end());
      double gpu_time = tim.lap_ms(); // get elapsed time

      double rate = (double)steps*(double)terms/(gpu_time*1000000.0);
      gpu_sum -= 0.5*(sinsum(0.0f,terms)+sinsum(pi,terms));
      gpu_sum *= step_size;
      printf("gpu_sum while loop sum = %.10f, steps %d terms %d time %.3f ms config %7d %4d rate %f \n",
        gpu_sum,steps,terms,gpu_time,blocks,threads,rate);
    };
    for ( int scale = 1; scale < 4096; scale <<= 1 ) {
      cl(scale);
    }
    return 0;
}