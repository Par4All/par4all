/* Free of a heap allocated area which has initialized pointer members
 * which are themselves heap allocated and have initialized pointer
 * members.
 *
 * Variation of pv_heap05: allocate an array of int**
*/

#include <stdlib.h>

int main()
{
  int ***p;
  int **q;
  int a = 0;
  q = (int **) malloc(sizeof(int*));
  q[0] = &a;
  p = (int ***) malloc(10*sizeof(int**));
  p[1] = q;
  free(p);
  return(0);
}
