/* free of a pointer through a different name than the one used 
 * for the allocation.
 *
 * Since the alias information is not preserved and since the heap is
 * represented with abstract values only, the value of p after the
 * call to free(q) is unknown.
 */
#include <stdlib.h>
int main()
{
  int *p, *q;
  int a;
  p = (int *) malloc(2*sizeof(int));
  p[0] = 0;
  p[1] = 1;
  q = p;
  free(q);
  return(0);
}
