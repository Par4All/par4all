#include <stdlib.h>
void * memload(void **dest, const void *src, size_t n) {
    size_t i;
    void *pdest = *dest=malloc(n);

    if( !pdest) return NULL;

    for(i=0;i<n;i++)
        ((char*)pdest)[i]=((const char*)src)[i];
    return pdest;
}
void * memstore(void **dest, const void *src, size_t n) {
    size_t i;
    void *pdest = *dest=malloc(n);

    if( !pdest) return NULL;

    for(i=0;i<n;i++)
        ((char*)pdest)[i]=((const char*)src)[i];
    return pdest;
}

void memalloc(void **ptr, size_t n) {
    *ptr=malloc(n);
}

void memfree(void *ptr) {
    free(ptr);
}
