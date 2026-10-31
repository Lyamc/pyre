// XZS step 1b: contention. (a) mapped zero-copy read (28x256) alone vs concurrent with a VRAM-heavy kernel on another stream;
// (b) CPU threads streaming DRAM (pool-like, 1.6 MB chunks) alone vs with concurrent zero-copy reads.
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <atomic>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#define CK(x) do{cudaError_t e=(x); if(e){printf("ERR %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} }while(0)
__global__ void rd4(const unsigned* __restrict__ p, size_t n4, unsigned* out, int reps){ unsigned acc=0; for(int r=0;r<reps;r++) for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n4;i+=(size_t)gridDim.x*blockDim.x){ acc^=p[i]+r; } if(acc==0x12345678) out[0]=acc; }
__global__ void vheavy(const uint4* __restrict__ p, size_t n16, unsigned* out, int reps){ unsigned acc=0; for(int r=0;r<reps;r++) for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<n16;i+=(size_t)gridDim.x*blockDim.x){ uint4 v=p[i]; acc=__dp4a(v.x,v.y,acc); acc=__dp4a(v.z,v.w,acc); } if(acc==0x12345678) out[0]=acc; }
int main(){
  const size_t EX=1587*1024; const int NE=12; size_t tot=EX*NE;
  CK(cudaSetDeviceFlags(cudaDeviceMapHost));
  char* h=(char*)malloc(tot+4096); char* ha=(char*)(((size_t)h+4095)&~(size_t)4095); for(size_t i=0;i<tot;i++) ha[i]=(char)i;
  CK(cudaHostRegister(ha,tot,cudaHostRegisterMapped)); void* dp; CK(cudaHostGetDevicePointer(&dp,ha,0));
  void* dv; CK(cudaMalloc(&dv,tot)); unsigned* out; CK(cudaMalloc(&out,4));
  cudaStream_t s1,s2; cudaStreamCreateWithFlags(&s1,cudaStreamNonBlocking); cudaStreamCreateWithFlags(&s2,cudaStreamNonBlocking);
  cudaEvent_t a1,b1,a2,b2; cudaEventCreate(&a1);cudaEventCreate(&b1);cudaEventCreate(&a2);cudaEventCreate(&b2); float m1,m2;
  for(int mode=0;mode<3;mode++){ float bz=1e9,bv=1e9; for(int r=0;r<5;r++){
    if(mode!=1){cudaEventRecord(a1,s1); rd4<<<28,256,0,s1>>>((const unsigned*)dp,tot/4,out,1); cudaEventRecord(b1,s1);}
    if(mode!=0){cudaEventRecord(a2,s2); vheavy<<<112,256,0,s2>>>((const uint4*)dv,tot/16,out,12); cudaEventRecord(b2,s2);}
    CK(cudaDeviceSynchronize()); if(mode!=1){cudaEventElapsedTime(&m1,a1,b1); if(m1<bz)bz=m1;} if(mode!=0){cudaEventElapsedTime(&m2,a2,b2); if(m2<bv)bv=m2;} }
    printf("mode %s: zc 12 experts %.1f us (%.2f GB/s)  vram-heavy %.1f us\n", mode==0?"zc alone  ":mode==1?"vram alone":"concurrent", mode==1?0:bz*1000, mode==1?0:tot/(bz*1e6), mode==0?0:bv*1000); }
  // CPU DRAM streaming
  const size_t CB=(size_t)4<<30; char* big=(char*)malloc(CB); for(size_t i=0;i<CB;i+=4096) big[i]=1;
  for(int nt: {16,24}) for(int zc=0;zc<2;zc++){
    std::atomic<bool> stop{false}; std::atomic<unsigned long long> bytes{0};
    std::vector<std::thread> th; for(int t=0;t<nt;t++) th.emplace_back([&,t]{ unsigned long long acc=0; size_t off=(size_t)t*EX*13 % (CB-EX); while(!stop){ const unsigned long long* p=(const unsigned long long*)(big+off); for(size_t i=0;i<EX/8;i+=1) acc+=p[i]; bytes+=EX; off=(off+EX*31)%(CB-EX);} if(acc==7)printf("x"); });
    std::this_thread::sleep_for(std::chrono::milliseconds(200)); unsigned long long b0=bytes; auto t0=std::chrono::steady_clock::now();
    int nk=0; if(zc){ auto te=t0+std::chrono::milliseconds(1000); while(std::chrono::steady_clock::now()<te){ cudaEventRecord(a1,s1); rd4<<<28,256,0,s1>>>((const unsigned*)dp,tot/4,out,1); cudaEventRecord(b1,s1); cudaEventSynchronize(b1); cudaEventElapsedTime(&m1,a1,b1); nk++; } } else std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    double dt=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count(); unsigned long long b1v=bytes; stop=true; for(auto&x:th)x.join();
    printf("cpu %d threads, zc %s: cpu %.1f GB/s%s", nt, zc?"on ":"off", (b1v-b0)/dt/1e9, zc?"":"\n"); if(zc) printf("  last zc 12 experts %.1f us (%.2f GB/s), %d kernels\n", m1*1000, tot/(m1*1e6), nk); }
  CK(cudaHostUnregister(ha)); return 0; }
