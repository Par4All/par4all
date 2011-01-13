#include "stars-pm-cuda.h"

//========================================================================
#define STRIDE_FORCE32 32
#define STRIDE_FORCE64 64
__global__ void k_forcex32(float *pot, float *fx)
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  float a,b,c;
  int idx=((bx*STRIDE_FORCE32-1)&(NP-1)) * NP * NP + by * NP + tx;
  a = pot[idx];
  idx=(idx+NP*NP)&(NP*NP*NP-1);
  b = pot[idx];

#pragma unroll
  for(int i=1; i<STRIDE_FORCE32+1;i++) {
    int oldIdx = idx;
    idx=(idx+NP*NP)&(NP*NP*NP-1);
    c = pot[idx];

    fx[oldIdx] = (c-a )/(2.*DX);

    a = b;
    b = c;

  }
}

__global__ void k_forcex64(float *pot, float *fx)
{
  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  float a,b,c;
  int idx=((bx*STRIDE_FORCE64-1)&(NP-1)) * NP * NP + by * NP + tx;
  a = pot[idx];
  idx=(idx+NP*NP)&(NP*NP*NP-1);
  b = pot[idx];

#pragma unroll
  for(int i=1; i<STRIDE_FORCE64+1;i++) {
    int oldIdx = idx;
    idx=(idx+NP*NP)&(NP*NP*NP-1);
    c = pot[idx];

    fx[oldIdx] = (c-a )/(2.*DX);

    a = b;
    b = c;

  }
}



void forcex(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  dim3 dimGridForceX32(NP/STRIDE_FORCE32,NP);
  dim3 dimGridForceX64(NP/STRIDE_FORCE64,NP);
  dim3 dimBlockForce(NP);

  if(NP<64) {
    P4A_launch_kernel(dimGridForceX32, dimBlockForce, k_forcex32, (float *)pot,(float *)fx);
  } else {
    P4A_launch_kernel(dimGridForceX64, dimBlockForce, k_forcex64, (float *)pot,(float *)fx);
  }
}


//========================================================================
__global__ void k_forcey(float *pot, float *fx)
{
  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  int cellCoord = bx * NP * NP + by*NP + tx;
  int prevCoord = cellCoord  - NP ;
  int nextCoord = cellCoord + NP;
  if(by==0) prevCoord += NP * NP;
  else if(by==NP-1) nextCoord -= NP*NP;

  __syncthreads();

  float x1 = pot[prevCoord];
  float x2 = pot[nextCoord];

  fx[cellCoord]=(x2-x1)/(2.*DX);
}

void forcey(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  dim3 dimGridForce(NP,NP);
  dim3 dimBlockForce(NP);
  P4A_launch_kernel(dimGridForce, dimBlockForce, k_forcey, (float *)pot,(float *)fx);
}


//========================================================================
__global__ void k_forcez( float *pot, float *fx )
{
  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  __shared__ float spot[NP];

  int blockCoord = bx * NP * NP + by*NP;
  int cellCoord = blockCoord + tx;

  spot[tx]=pot[cellCoord];

  __syncthreads();

  float x1 = spot[ ((unsigned int)(tx - 1 )) % NP ];
  float x2 = spot[ (tx + 1 ) % NP ];

  fx[cellCoord]=(x2-x1)/(2.*DX);
}

void forcez(float pot[NP][NP][NP], float fx[NP][NP][NP]) {
  dim3 dimGridForce(NP,NP);
  dim3 dimBlockForce(NP);
  P4A_launch_kernel(dimGridForce, dimBlockForce, k_forcez, (float *)pot,(float *)fx);
}



/************ OLDIES **************************


//========================================================================
__global__ void carte_force( float *pot, float *fx, int coord )
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  __shared__ float spot[NP+2];

  // FIXME bricolage...
  int coords[3][3];
  coords[0][0]=bx + by*NP + tx*NP*NP;
  coords[0][1]=bx + tx*NP + by*NP*NP;
  coords[0][2]=tx + bx*NP + by*NP*NP;
  coords[1][0]=bx + by*NP + NP*NP*(NP-1);
  coords[1][1]=bx + (NP-1)*NP + NP*NP*by;
  coords[1][2]=(NP-1) + bx*NP + NP*NP*by;
  coords[2][0]=bx + by*NP + NP*NP*0;
  coords[2][1]=bx + 0*NP + NP*NP*by;
  coords[2][2]=0 + bx*NP + NP*NP*by;


  spot[tx+1]=pot[coords[0][coord]];
  spot[0]=pot[coords[1][coord]];
  spot[NP+1]=pot[coords[2][coord]];

  __syncthreads();

  fx[coords[0][coord]]=(spot[tx+2]-spot[tx])/(2.*DX);

}

//========================================================================
__global__ void new_carte_forcex( float *pot, float *fx, int coord )
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  // FIXME bricolage

  int cellCoord = bx * NP * NP + by * NP + tx;
  int prevCoord = cellCoord  - NP * NP ;
  int nextCoord = cellCoord + NP * NP;
  if(bx==0) prevCoord += NP * NP * NP;
  else if(bx==NP-1) nextCoord -= NP * NP * NP;

  __syncthreads();

  float x1 = pot[prevCoord];
  float x2 = pot[nextCoord];

  fx[cellCoord]=(x2-x1)/(2.*DX);

}
//========================================================================
__global__ void new_carte_forcex2( float *pot, float *fx, int coord )
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  // FIXME bricolage

  int cellCoord = bx * NP * NP + by * NP + tx;
  int prevCoord = (((unsigned int)(bx - 1)) % NP ) * NP * NP + by * NP + tx;;
  int nextCoord = (((unsigned int)(bx + 1)) % NP ) * NP * NP + by * NP + tx;;;

  __syncthreads();

  float x1 = pot[prevCoord];
  float x2 = pot[nextCoord];

  fx[cellCoord]=(x2-x1)/(2.*DX);

}
//========================================================================
__global__ void carte_forcex( float *pot, float *fx, int coord )
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  __shared__ float spot[NP+2];

  // FIXME bricolage...
  int firstCellCoord = bx + by*NP;
  int cellCoord = firstCellCoord + tx*NP*NP;

  float *row=pot + tx*NP*NP;
  float *lastrow= pot + (NP-1)*NP*NP;

  spot[tx+1]=row[firstCellCoord]/(2.*DX);
  spot[0]=lastrow[firstCellCoord]/(2.*DX);
  spot[NP+1]=pot[firstCellCoord]/(2.*DX);

  __syncthreads();

  fx[cellCoord]=spot[tx+2]-spot[tx];

}

//========================================================================
__global__ void carte_forcex_old(float *pot, float *fx)
{

  int tx=threadIdx.x;

  int bx=blockIdx.x;
  int by=blockIdx.y;

  __shared__ float spot[NP+2];
  __shared__ float sfx[NP];

  spot[tx+1]=pot[bx+by*NP+NP*NP*tx];

  spot[0]=pot[bx+by*NP+NP*NP*(NP-1)];
  spot[NP+1]=pot[bx+by*NP+NP*NP*0];

  __syncthreads();

  sfx[tx]=(spot[tx+2]-spot[tx])/2./DX;

  fx[bx+by*NP+NP*NP*tx]=sfx[tx];

}

//========================================================================
__global__ void new_carte_forcey(float *pot, float *fx)
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  int cellCoord = bx * NP * NP + by*NP + tx;
  int prevCoord = cellCoord  - NP ;
  int nextCoord = cellCoord + NP;
  if(by==0) prevCoord += NP * NP;
  else if(by==NP-1) nextCoord -= NP*NP;

  __syncthreads();

  float x1 = pot[prevCoord];
  float x2 = pot[nextCoord];

  fx[cellCoord]=(x2-x1)/(2.*DX);
}
__global__ void new_carte_forcey2(float *pot, float *fx)
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  int cellCoord = bx * NP * NP + by*NP + tx;
  int prevCoord = bx * NP * NP + (((unsigned int)(by-1))%NP) * NP + tx;
  int nextCoord = bx * NP * NP + (((unsigned int)(by+1))%NP) * NP + tx;

  __syncthreads();

  float x1 = pot[prevCoord];
  float x2 = pot[nextCoord];

  fx[cellCoord]=(x2-x1)/(2.*DX);
}
//========================================================================
__global__ void old_carte_forcey(float *pot, float *fx)
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  __shared__ float spot[NP+2];
  __shared__ float sfx[NP];

  spot[tx+1]=pot[bx+tx*NP+NP*NP*by];

  spot[0]=pot[bx+(NP-1)*NP+NP*NP*by];
  spot[NP+1]=pot[bx+0*NP+NP*NP*by];

  __syncthreads();

  sfx[tx]=(spot[tx+2]-spot[tx])/(2.*DX);

  fx[bx+tx*NP+NP*NP*by]=sfx[tx];

}

//========================================================================
__global__ void carte_forcey( float *pot, float *fx, int coord )
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  __shared__ float spot[NP+2];

  // FIXME bricolage...

  int cellCoord=bx + tx*NP + by*NP*NP;
  int firstCellCoord=bx + 0*NP + NP*NP*by;
  int lastCellCoord=bx + (NP-1)*NP + NP*NP*by;


  spot[tx+1]=pot[cellCoord];
  spot[0]=pot[firstCellCoord];
  spot[NP+1]=pot[lastCellCoord];

  __syncthreads();

  fx[cellCoord]=(spot[tx+2]-spot[tx])/(2.*DX);
}

//========================================================================
__global__ void carte_forcez( float *pot, float *fx, int coord)
{

  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  __shared__ float spot[NP];

  int blockCoord = bx * NP * NP + by*NP;
  int cellCoord = blockCoord + tx;

  spot[tx]=pot[cellCoord];

  __syncthreads();

  float x1 = spot[ ((unsigned int)(tx - 1 )) % NP ];
  float x2 = spot[ (tx + 1 ) % NP ];

  fx[cellCoord]=(x2-x1)/(2.*DX);

}
//========================================================================
__global__ void old_carte_forcez(float *pot, float *fx)
{
  int tx=threadIdx.x;
  int bx=blockIdx.x;
  int by=blockIdx.y;

  __shared__ float spot[NP+2];
  __shared__ float sfx[NP];

  spot[tx+1]=pot[tx+bx*NP+NP*NP*by];

  spot[0]=pot[NP-1+bx*NP+NP*NP*by];
  spot[NP+1]=pot[0+bx*NP+NP*NP*by];

  __syncthreads();

  sfx[tx]=(spot[tx+2]-spot[tx])/2./DX;

  fx[tx+bx*NP+NP*NP*by]=sfx[tx];

}
******************************************/
