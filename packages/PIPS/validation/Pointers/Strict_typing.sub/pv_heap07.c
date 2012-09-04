/* free of a pointer through a different name than the one used 
   for the allocation */
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
