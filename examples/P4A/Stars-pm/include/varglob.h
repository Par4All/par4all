#ifndef __VARGLOB__H_
#define __VARGLOB__H_

#ifndef NP
// Default size
#define NP 128
#endif

#define NPART (NP*NP*NP)
#define LBOX 6.
#define G 1.
#define TMAX 7.5
#define DX (LBOX/NP)
#define DT (5*1e-2)
#define NPBLOCK 128
#define BLOCK_SIZE 2
#define MODT 1000
#define MODDISP 4
#define CUERR() printf("\n %s \n",cudaGetErrorString(cudaGetLastError()))

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif


typedef struct coord {
  float _[3];
} coord;

#endif // __VARGLOB__H_
