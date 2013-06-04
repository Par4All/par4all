#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>

//#include "alloc_nrc.h"
//#include "alloc.h"
//#include "alloc_rgb.h"

typedef unsigned char byte;
typedef struct{byte r; byte g; byte b;} rgb8;

// -------------------------------------------------------
float** matrix_nrc(long nrl, long nrh, long ncl, long nch)
// -------------------------------------------------------
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
    float **m;
    
    /* allocate pointers to rows */
    m=(float **) malloc((size_t)((nrow)*sizeof(float*)));
    //if (!m) nrerror("allocation failure 1 in matrix()");
    m -= nrl;
    
    /* allocate rows and set pointers to them */
    m[nrl]=(float *) malloc((size_t)((nrow*ncol)*sizeof(float)));
    //if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
    m[nrl] -= ncl;
    
    for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;
    
    /* return pointer to array of pointers to rows */
    return m;
}
