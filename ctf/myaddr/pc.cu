#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE (1<<5)
__global__ void fetch(unsigned long *d)
{
  int x = threadIdx.x;
  unsigned long s = 0x61L;
  d[x] = s;
  __syncthreads();
}

#include <string>
#include <iostream>

static const char hexes[] = "0123456789ABCDEF";

void HexDump(FILE *f, const unsigned char *From, int Len)
{
 int i;
 int j,k;
 char buffer[256];
 char *ptr;

 for(i=0;i<Len;)
     {
          ptr = buffer;
          sprintf(ptr, "%08X ",i);
          ptr += 9;
          for(j=0;j<16 && i<Len;j++,i++)
          {
             *ptr++ = j && !(j%4)?(!(j%8)?'|':'-'):' ';
             *ptr++ = hexes[From[i] >> 4];
             *ptr++ = hexes[From[i] & 0xF];
          }
          for(k=16-j;k!=0;k--)
          {
            ptr[0] = ptr[1] = ptr[2] = ' ';
            ptr += 3;

          }
          ptr[0] = ptr[1] = ' ';
          ptr += 2;
          for(;j!=0;j--)
          {
               if(From[i-j]>=0x20 && From[i-j]<0x80)
                    *ptr = From[i-j];
              else
                    *ptr = '.';
               ptr++;
          }
          *ptr = 0;
          fprintf(f, "%s\n", buffer);
     }
     fprintf(f, "\n");
}

// main
__host__ int main()
{
  unsigned long *d_i;
  unsigned long host_d[2];
  memset(host_d, 0xcc, sizeof(host_d));
  cudaMalloc(&d_i, sizeof(host_d));
  printf("d_i %p\n", d_i);
  fetch<<<1,1>>>(d_i);
  cudaMemcpy(host_d, d_i, sizeof(host_d), cudaMemcpyDeviceToHost);
  cudaFree(d_i);
  if ( host_d[0] ) printf("from device %lx\n", host_d[0]);
  HexDump(stdout, (const unsigned char *)host_d, sizeof(host_d));
}