/* free of a heap allocated area which has initialized pointer members
 * which are themselves heap allocated and have initialized pointer
 * members
 *
 * Variation on pv_heap05: the indexing p[1] is wrong.
*/
#include <stdlib.h>
int main()
{
  int ***p;
  int **q;
  int a = 0;
  q = (int **) malloc(sizeof(int*));
  q[0] = &a;
  p = (int ***) malloc(sizeof(int**));
  p[1] = q;
  free(p);
  return(0);
}
