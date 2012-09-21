#include <stdio.h>
#include <stdlib.h>

typedef struct {int dim; float * data;} darray_t, * parray_t;

void allocate_array(parray_t pa) {
  pa->data = (float *) malloc(pa->dim*sizeof(float));
  return;
}

void init_array(parray_t pa) {
  int i;
  for(i=0; i<pa->dim; i++)
    pa->data[i] = (float) i;

  return;
}

int main() {
  parray_t ma = (parray_t) malloc(sizeof(darray_t));
  allocate_array(ma);
  //init_array(ma);

  return 0;
}
