typedef struct farray_t * farray;

farray farray_new(unsigned int n);
void farray_delete(farray a);
float farray_get(farray a,unsigned int i);
void farray_set(farray a,unsigned int i,float f);
