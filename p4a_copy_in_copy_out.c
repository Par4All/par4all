#include <stdlib.h>
void * p4a_copy_in(void **dest, const void *src, size_t n) {
    size_t i;
    void *pdest = *dest=malloc(n);

    if( !pdest) return NULL;

    for(i=0;i<n;i++)
        ((char*)pdest)[i]=((const char*)src)[i];
    return pdest;
}
void * p4a_copy_out(void **dest, const void *src, size_t n) {
    size_t i;
    void *pdest = *dest=malloc(n);

    if( !pdest) return NULL;

    for(i=0;i<n;i++)
        ((char*)pdest)[i]=((const char*)src)[i];
    return pdest;
}

void * p4a_allocate(void **dest, const void *src, size_t n) {
    size_t i;
    void *pdest = *dest=malloc(n);

    if( !pdest) return NULL;

    for(i=0;i<n;i++)
        ((char*)pdest)[i]=((const char*)src)[i];
    return pdest;
}
