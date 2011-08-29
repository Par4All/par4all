#include "array.h"
struct farray_t
{
    unsigned int size;
    float *data;
};

farray farray_new(unsigned int n)
{
    farray a ;
    a = (farray)malloc(sizeof(*a));
    if( a ) {
        a->size = n;
        a->data = (float*)malloc(n*sizeof(*(a->data)));
        if( a->data )
            return a;
        else
            free(a);
    }
    return 0;
}

void farray_delete(farray a)
{
    free(a->data);
    free(a);
}

float farray_get(farray a,unsigned int i)
{
    return a->data[i];
}
void farray_set(farray a,unsigned int i,float f)
{
    a->data[i]=f;
}
