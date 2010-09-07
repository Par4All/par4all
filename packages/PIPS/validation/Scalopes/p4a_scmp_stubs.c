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
		   size_t n) {
    memcpy(in_address, out_address, n);
}


void P4A_scmp_write(void * out_address,
			   const void * buffer_address,
			   size_t n) {
    memcpy(out_address, buffer_address, n);
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
