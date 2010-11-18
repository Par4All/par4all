#ifndef __VARGLOB__H_
#define __VARGLOB__H_

#define NCELL 128
#define NPART (NCELL*NCELL*NCELL)
#define NELEM (NPART)
#define LBOX 6.
#define G 1.
#define TMAX 10
#define DX (LBOX/NCELL)
#define DT (5*1e-2)
#define BLOCK_SIZE 2
#define NPBLOCK (NCELL)
#define NTHREAD 256
#define NDATA_PER_THREAD 1
#define MAXDIMZ 32
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
