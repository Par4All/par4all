/* To investigate a bug in tableau_dynamique
 *
 * A recursive descent in dereferencing.c is required for lhs pa->data[i]
 *
 * Then effects computation fails because of "potential memory
 * overflow due to effect" in
 * simple_cell_reference_with_address_of_cell_reference_translation()
 */

typedef struct {int dim; float * data;} darray_t, * parray_t;

void init_array(parray_t pa) {
  int i;
  for(i=0; i<pa->dim; i++)
    pa->data[i] = (float) i;

  return;
}
