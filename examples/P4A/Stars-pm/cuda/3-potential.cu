#include <math.h>
#include <cufft.h>

#include "stars-pm-cuda.h"

#define MAXDIMZ 32

__global__ void k_int2float(int *v1, float *v2) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;

  v2[bx * NPBLOCK + tx] = v1[bx * NPBLOCK + tx] - 1;
}

__global__ void k_float2int(float *v1, int *v2) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;

  v2[bx * NPBLOCK + tx] = v1[bx * NPBLOCK + tx];
}

__global__ void k_real2Complex(cufftReal *rfield, cufftComplex *field) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;

  field[bx * NP * NP + by * NP + tx] = make_cuFloatComplex(rfield[bx * NP * NP
      + by * NP + tx], 0.);
}

__global__ void k_complex2Real(cufftComplex *field, cufftReal *rfield) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;

  rfield[bx * NP * NP + by * NP + tx] = cuCrealf(field[bx * NP * NP + by * NP
      + tx]);
}

__global__ void k_complex2Real_correctionPot(cufftComplex *field, cufftReal *rfield, float coeff) {
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int gIdx = bx * NP * NP + by * NP + tx;
  rfield[gIdx] = cuCrealf(field[gIdx]) * coeff / (DX * DX * DX);
}


__global__ void k_correction_pot(float *pot, float coeff) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int gIdx = bx * NP * NP + by * NP + tx;

  pot[gIdx] = pot[gIdx] * coeff / (DX * DX * DX);
}


__global__ void k_fft_laplacian7(cufftComplex *field) {
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;

  __shared__ float i2,j2;
  float k2;

  int offset;
  int limit = NP >> 2;
  float coeff = M_PI / NP;
  if(k == 0) {
    if(i > limit) {
      offset = NP;
    } else {
      offset = 0;
    }
    i2 = sinf(coeff * (i - offset));

    offset = 0;
    if(j > limit) {
      offset = NP;
    } else {
      offset = 0;
    }
    j2 = sinf(coeff * (j - offset));

    offset = 0;
  } else {
    if(k > limit) {
      offset = NP;
    } else {
      offset = 0;
    }
  }

  k2 = sinf(coeff * (k - offset));
  k2 = k2 * k2;
  __syncthreads();

  k2 = k2 + i2 * i2 + j2 * j2;
  k2 += (k2 == 0);

  int gIdx = i * NP * NP + j * NP + k;
  cufftComplex _field = field[gIdx];
  coeff = G *M_PI * DX * DX / k2 / NP / NP / NP;
  field[gIdx] = make_cuFloatComplex(cuCrealf(_field) * coeff, cuCimagf(_field) * coeff); // FFT NORMALISATION
}

static cufftHandle plan = NULL;
cufftComplex *cdens = NULL;

void potential_init_plan(float unused[NP][NP][NP][2]) {
  cufftPlan3d(&plan, NP, NP, NP, CUFFT_C2C);
  toolTestExec(cudaMalloc((void**)&cdens,sizeof(cufftComplex)*NP*NP*NP));
}

void potential_free_plan() {
}

void potential(int histo[NP][NP][NP],
               float dens[NP][NP][NP],
               float unused[NP][NP][NP][2],
               float mp[NP][NP][NP]) {
  dim3 dimGridC2R(NP, NP);
  dim3 dimBlockC2R(NP);

  dim3 dimGrid(NP, NP);
  dim3 dimBlock(NP);

  dim3 dimGriddata(NPART / NPBLOCK);
  dim3 dimBlockdata(NPBLOCK);

  P4A_launch_kernel(dimGriddata,dimBlockdata,k_int2float,(int *)histo,(float *)dens);

  P4A_launch_kernel(dimGridC2R,dimBlockC2R,k_real2Complex,(cufftReal *)dens,(cufftComplex *)cdens);

  cufftExecC2C(plan,
               (cufftComplex *)cdens,
               (cufftComplex *)cdens,
               CUFFT_FORWARD);

  P4A_launch_kernel(dimGrid,dimBlock,k_fft_laplacian7,(cufftComplex *)cdens);

  cufftExecC2C(plan,
               (cufftComplex *)cdens,
               (cufftComplex *)cdens,
               CUFFT_INVERSE);

  P4A_launch_kernel(dimGridC2R,dimBlockC2R,k_complex2Real_correctionPot,(cufftComplex *)cdens,(float *)dens, mp[0][0][0]);
/*
  Fused is 2 times faster
  P4A_launch_kernel(dimGridC2R,dimBlockC2R,k_complex2Real,(cufftComplex *)cdens,(float *)dens);
  P4A_launch_kernel(dimGridC2R,dimBlockC2R,k_correction_pot, (float *)dens, mp[0][0][0]);
*/
}

