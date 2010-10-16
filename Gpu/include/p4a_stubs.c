#include <stdlib.h>
void * P4A_copy_from_accel(void *dest, const void *src, size_t size,
        size_t d1_length,
        size_t d1_offset,
        size_t d1_block_length) {
    size_t i,l;
    char * cdest = (char*)dest,
         *csrc = (char*)src;
    for(i=0;i<d1_block_length;i++)
        for(l=0;l<size;l++)
            cdest[(i+d1_offset)*size+l]=csrc[i*size+l];
    return dest;
}
void * P4A_copy_to_accel(void *dest, const void *src, size_t size,
        size_t d1_length,
        size_t d1_offset,
        size_t d1_block_length) {
    size_t i,l;
    char * cdest = (char*)dest,
         *csrc = (char*)src;
    for(i=0;i<d1_block_length;i++)
        for(l=0;l<size;l++)
            cdest[i*size+l]=csrc[(i+d1_offset)*size+l];
    return dest;
}
void * P4A_copy_from_accel2d(void *dest, const void *src, size_t size,
        size_t d1_length, size_t d2_length,
        size_t d1_offset, size_t d2_offset,
        size_t d1_block_length, size_t d2_block_length)
{
    size_t i,j,l;
    char * cdest = (char*)dest,
         *csrc = (char*)src;
    for(i=0;i<d1_block_length;i++)
        for(j=0;j<d2_block_length;j++)
                for(l=0;l<size;l++)
                    cdest[(i+d1_offset)*size*d2_length+((j+d2_offset)*size+l) ]=
                        csrc[i*size*d2_block_length+j*size+l];
    return dest;
}
void * P4A_copy_to_accel2d(void *dest, const void *src, size_t size,
        size_t d1_length, size_t d2_length,
        size_t d1_offset, size_t d2_offset,
        size_t d1_block_length, size_t d2_block_length) {
    size_t i,j,l;
    char * cdest = (char*)dest,
         *csrc = (char*)src;
    for(i=0;i<d1_block_length;i++)
        for(j=0;j<d2_block_length;j++)
                for(l=0;l<size;l++)
                    cdest[i*size*d2_block_length+j*size+l ]=
                        csrc[(i+d1_offset)*size*d2_length+(j+d2_offset)*size+l];
    return dest;
}

void P4A_accel_malloc(void **ptr, size_t n) {
    if(n) *ptr=malloc(n);
    else *ptr=NULL;
}

void P4A_accel_free(void **ptr) {
    free(*ptr);
    *ptr=NULL;
}
