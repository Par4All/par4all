#include<stdlib.h>

char** bmatrix(long nrl, long nrh, long ncl, long nch)
/* ------------------------------------------------ */
/* allocate an uchar matrix with subscript range m[nrl..nrh][ncl..nch] */
{
  long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
  char **m;

  /* allocate pointers to rows */
  m = (char **) malloc((size_t)((nrow+ncol)*sizeof(char*)));
  for(i=nrl+1;i<=nrh;i++)
    m[i] = m[i-1]+ncol;
  /* return pointer to array of pointers to rows */
  return m;
 
}
