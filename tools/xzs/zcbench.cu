// zero-copy PCIe read microbench (pyre XZS step 1). Reads expert-sized (1.587 MB) blocks from mapped pinned host memory.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
__global__ void rd16(const uint4* __restrict__ p, size_t n16, unsigned* out){ unsigned acc=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n16;i+=(size_t)gridDim.x*blockDim.x){ uint4 v=p[i]; acc^=v.x^v.y^v.z^v.w; } if(acc==0x12345678) out[0]=acc; }
__global__ void rd4(const unsigned* __restrict__ p, size_t n4, unsigned* out){ unsigned acc=0; for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n4;i+=(size_t)gridDim.x*blockDim.x){ acc^=p[i]; } if(acc==0x12345678) out[0]=acc; }
int main(){
  const size_t EX=1587*1024; const int NE=16; size_t tot=EX*NE;
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  char* h=(char*)malloc(tot+4096); char* ha=(char*)(((size_t)h+4095)&~(size_t)4095); for(size_t i=0;i<tot;i++) ha[i]=(char)i;
  CK(cudaHostRegister(ha,tot,cudaHostRegisterMapped)); void* dp; CK(cudaHostGetDevicePointer(&dp,ha,0));
  void* dv; CK(cudaMalloc(&dv,tot)); CK(cudaMemcpy(dv,ha,tot,cudaMemcpyHostToDevice)); unsigned* out; CK(cudaMalloc(&out,4));
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b); float ms;
  int nes[]={1,4,12,16}; int grids[]={28,56,112,224};
  for(int src=0;src<3;src++){ for(int ne: nes){ for(int g: grids){ for(int w=0;w<2;w++){
    const void* p = src==2? dv : dp; size_t bytes=EX*ne;
    float best=1e9; for(int r=0;r<5;r++){ cudaEventRecord(a); if(w==0) rd16<<<g,256>>>((const uint4*)p,bytes/16,out); else rd4<<<g,256>>>((const unsigned*)p,bytes/4,out); cudaEventRecord(b); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&ms,a,b); if(ms<best) best=ms; }
    if(src==1) continue;
    printf("%s ld%d experts=%2d grid=%3d : %8.1f us  %6.2f GB/s  per-expert %6.1f us\n", src==2?"VRAM  ":"MAPPED", w?4:16, ne, g, best*1000, bytes/(best*1e6), best*1000/ne); }}}}
  // DMA reference
  float best=1e9; for(int r=0;r<5;r++){ cudaEventRecord(a); cudaMemcpyAsync(dv,ha,EX*12,cudaMemcpyHostToDevice); cudaEventRecord(b); cudaEventSynchronize(b); cudaEventElapsedTime(&ms,a,b); if(ms<best)best=ms; }
  printf("DMA H2D 12 experts: %.1f us %.2f GB/s\n",best*1000,EX*12/(best*1e6));
  CK(cudaHostUnregister(ha)); return 0; }
