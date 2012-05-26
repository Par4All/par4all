#include <complex.h>

//#include "mpi.h"
/* pseudo mpi.h file */

#define MPI_DOUBLE 1
#define MPI_SUM 1
#define MPI_COMM_WORLD 1
#define MPI_INT 1

#define NMS (100)
static int nms=NMS,nvhs;
//static complex **vs,**zvs;

// allocate the arrays
static complex array_vs[NMS][NMS], array_zvs[NMS][NMS];
static double complex array_avd[NMS][NMS], scalar_cs1, scalar_cs2;

// Initialize the pointers used by the application
static complex (*vs)[NMS][NMS] = &array_vs;
static complex (*zvs)[NMS][NMS] = &array_zvs;
static double complex **vds,**zvds, (*avd)[NMS][NMS] = &array_avd,
  *cs1 = &scalar_cs1,*cs2 = &scalar_cs2;

void sum_vprod(int n)
{
  int i;

  if ((4*8*8*4)>1)
    {
      /* MPI_Reduce((double*)(cs1),(double*)(cs2),2*n,1, */
      /* 		 1,0,1); */
      /* MPI_Bcast((double*)(cs2),2*n,1,0,1); */
    }
  else
    {
      for (i=0;i<n;i++)
	{
	  cs2[i]=cs1[i];
	}
    }
}

// Transposed version of cmat_vec_dble using the native complex type (Petaqcd)

void cmat_vec_dble(int n, double complex a[n][n], double complex v[n], double complex w[n])
{
  int i, j;

  for(i=0;i<n;i++) {
      w[i] = 0.0;
      for(j=0;j<n;j++) {
	w[i] += a[i][j] * v[j];
      }
  }
}

/* Compute z = v * conj(w) */
complex vprod(int n,int icom,complex v[n],complex w[n])
{
   complex z;
   double complex vd,wd;
   int i;

   vd=0.0;

   for (i=0;i<n;i++)
   {
     /* Produit par le conjugue? */
     vd += v[i]*conj(w[i]);
   }

   if ((icom!=1)||((4*8*8*4)==1))
   {
      z=(complex)wd;
   }
   else
   {
     double x = cdouble(vd);
     double y = cdouble(wd);
      MPI_Reduce(&x,&y,2,1,1,0,1);
      MPI_Bcast(&y,2,1,0,1);

      wd = (complex) y;
      z=(complex)wd;
   }

   return z;
}

/* Compute v = z * w */
void mulc_vadd(int n,complex v[n],complex w[n],complex z)
{
  int i;
  for (i=0; i<n;i++)
    {
      v[i] += z*w[i];
    }
}

void Lv(complex v[nms])
{
  if(1) {
   int nm, nvh, i;
   complex z;

   nm = nms;
   nvh = nvhs;

   for(i = 0; i <= nm-1; i += 1) {
     z = vprod(nvh, 0, (*vs)[i], v);
     cs1[i] = (double complex) z;
   }

   sum_vprod(nm);
   cmat_vec_dble(nm, * avd, cs2, cs1);

   for(i = 0; i <= nm-1; i += 1) {
     z = -((float) cs1[i]);
     mulc_vadd(nvh, v, (*zvs)[i], z);
   }
  }
}
