#include <math.h>
#include <cufft.h>

#include "stars-pm-cuda.h"



__global__ void k_int2float(int *v1, float *v2) {
  int tx=threadIdx.x;
  int bx=blockIdx.x;

  v2[bx*NPBLOCK+tx]=v1[bx*NPBLOCK+tx]-1;
}

__global__ void k_real2Complex(cufftReal *rfield, cufftComplex *field) {
  int bx=blockIdx.x;
  int by=blockIdx.y;
  int tx=threadIdx.x;

  field[bx*NP*NP+by*NP+tx]=make_cuFloatComplex(rfield[bx*NP*NP+by*NP+tx],0.);
}

__global__ void k_complex2Real(cufftComplex *field, cufftReal *rfield)
{
  int bx=blockIdx.x;
  int by=blockIdx.y;
  int tx=threadIdx.x;

  rfield[bx*NP*NP+by*NP+tx]=cuCrealf(field[bx*NP*NP+by*NP+tx]);
}


__global__ void k_fft_laplacian7(cufftComplex *field)
{
  int tx=threadIdx.x;
  int ty=threadIdx.y;
  int tz=threadIdx.z;

  int bx=blockIdx.x;
  int by=blockIdx.y;

  int j=bx*BLOCK_SIZE+tx ;
  int i=by*BLOCK_SIZE+ty ;
  int k;
  int koffset;

  float i2,j2,k2;

  __shared__ cufftComplex shared_field[BLOCK_SIZE][BLOCK_SIZE][MAXDIMZ];

  i2= (i>NP/2.?sinf(M_PI/NP*(i-NP))*sinf(M_PI/NP*(i-NP)):sinf(M_PI/NP*(i))*sinf(M_PI/NP*(i)));
  j2= (j>NP/2.?sinf(M_PI/NP*(j-NP))*sinf(M_PI/NP*(j-NP)):sinf(M_PI/NP*(j))*sinf(M_PI/NP*(j)));

  for(koffset=0;koffset<=NP/MAXDIMZ-1;koffset++) {
    k=koffset*MAXDIMZ+tz;

    shared_field[tx][ty][tz]=field[i*NP*NP+j*NP+k];

    k2= (k>NP/2.?sinf(M_PI/NP*(k-NP))*sinf(M_PI/NP*(k-NP)):sinf(M_PI/NP*k)*sinf(M_PI/NP*k))+i2+j2;
    k2+=(k2==0);

    __syncthreads();
    field[i*NP*NP+j*NP+k]=make_cuFloatComplex(cuCrealf(shared_field[tx][ty][tz])*G*M_PI*DX*DX/k2/NP/NP/NP,cuCimagf(shared_field[tx][ty][tz])*G*M_PI*DX*DX/k2/NP/NP/NP); // FFT NORMALISATION
//    field[0]=make_cuFloatComplex(0.,0.);

    __syncthreads();

  }
}

__global__ void k_correction_pot(float *pot, float coeff)
{
  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  int cellCoord = bx * NP * NP + by * NP + tx;


  pot[cellCoord]=(float) (pot[cellCoord])*coeff/(DX*DX*DX);
}



static cufftHandle plan = NULL;
cufftComplex *cdens = NULL;

void potential_init_plan(float unused[NP][NP][NP][2]) {
  cufftPlan3d(&plan,NP,NP,NP,CUFFT_C2C);
  toolTestExec(cudaMalloc((void**)&cdens,sizeof(cufftComplex)*NP*NP*NP));
}

void potential_free_plan() {
}


void potential(int histo[NP][NP][NP],
               float dens[NP][NP][NP],
               float unused[NP][NP][NP][2],
               float mp[NP][NP][NP] ) {
  dim3 dimGridC2R(NP,NP);
  dim3 dimBlockC2R(NP);

  dim3 dimGrid(NP/BLOCK_SIZE,NP/BLOCK_SIZE);
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,MAXDIMZ);

  dim3 dimGriddata(NPART/NPBLOCK);
  dim3 dimBlockdata(NPBLOCK);

  P4A_launch_kernel(dimGriddata,dimBlockdata,k_int2float,(int *)histo,(float *)dens);

  P4A_launch_kernel(dimGridC2R,dimBlockC2R,k_real2Complex,(cufftReal *)dens,(cufftComplex *)cdens);

  cufftExecC2C(plan,(cufftComplex *)cdens,(cufftComplex *)cdens,CUFFT_FORWARD);

  P4A_launch_kernel(dimGrid,dimBlock,k_fft_laplacian7,(cufftComplex *)cdens);

  cufftExecC2C(plan,(cufftComplex *)cdens,(cufftComplex *)cdens,CUFFT_INVERSE);

  P4A_launch_kernel(dimGridC2R,dimBlockC2R,k_complex2Real,(cufftComplex *)cdens,(float *)dens);

  P4A_launch_kernel(dimGridC2R,dimBlockC2R,k_correction_pot, (float *)dens, mp[0][0][0]);

}

