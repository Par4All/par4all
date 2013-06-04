#include<stdlib.h>
#include <stdio.h>
#define NR_END 1
#define FREE_ARG char*


// ---------------------------------------------------
float** matrix(long nrl, long nrh, long ncl, long nch)
// ---------------------------------------------------
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
    long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
    float **m;
    
    /* allocate pointers to rows */
    m=(float **) malloc((size_t)((nrow)*sizeof(float*)));
    //if (!m) nrerror("allocation failure 1 in matrix()");
    m -= nrl;
    
    for(i=nrl;i<=nrh;i++) {
        m[i] = malloc((size_t)(ncol)*sizeof(float));
        //if (!m[if]) nrerror("allocation failure 2 in matrix()");
        m[i] -= ncl;
    }
    /* return pointer to array of pointers to rows */
    return m;
}


