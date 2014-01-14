/* To investigate a bug in for condition analysis
 *
 * Derived from init_array
 */

typedef struct {int dim; float * data;} darray_t, * parray_t;

void init_array02(parray_t pa) {
  int i, j=0;
  for(i=0; i<pa->dim; i++)
    j++;

  return;
}
