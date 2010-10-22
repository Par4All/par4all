#include <stdlib.h>
#include <string.h>

typedef struct dtable{
  int ** buffers;
  int size;
}dtable;

dtable * buffertable=NULL;


/*VERY QUICK AND VERY DIRTY!!*/

void P4A_scmp_read(const int * in_address,
		   int * out_address,
           size_t type_size,
		   size_t nmemb,
           size_t offset,
           size_t dim1_size) {

    memcpy(in_address, out_address+offset*type_size, nmemb*type_size);
}

void P4A_scmp_write(void * out_address,
			   const void * buffer_address,
               size_t type_size,
			   size_t nmemb,
               size_t offset,
               size_t dim1_size) {
    memcpy(out_address+offset*type_size, buffer_address, nmemb*type_size);
}

void P4A_scmp_malloc(void **dest,  size_t n) {

  if(*dest==(void*)0)
      *dest = malloc(n);
}


void P4A_scmp_dealloc(void *dest) {
  /* free(dest);	 */
}

void * P4A_scmp_flow(void** flow) {
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
