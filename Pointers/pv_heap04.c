/* free of a heap allocated area which has initialized pointer members */
#include <stdlib.h>
int main()
{
  int **p;
  int *q;
  int a = 0;
  q = (int *) malloc(sizeof(int));
  q[0] = a;
  p = (int **) malloc(sizeof(int*));
  p[0] = q;
  free(p);
  return(0);
}
