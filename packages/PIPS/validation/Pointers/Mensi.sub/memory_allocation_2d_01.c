/* float32 replaced by float  
   nrerror replaced by assert
*/
#include<stdlib.h>
#include <assert.h>

/* ----------------------------------------------------- */
float** f32matrix(long nrl, long nrh, long ncl, long nch)
/* ----------------------------------------------------- */
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  float **m;

  /* allocate pointers to rows */
  m=(float **) malloc((size_t)((nrow)*sizeof(float*)));
  if (!m) assert("allocation failure 1 in f32matrix()");
  m -= nrl;

  /* allocate rows and set pointers to them */
  m[nrl]=(float *) malloc((size_t)((nrow*ncol)*sizeof(float)));
  if (!m[nrl]) assert("allocation failure 2 in f32matrix()");
  m[nrl] -= ncl;

  for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

  /* return pointer to array of pointers to rows */
  return m;
}
