/* malloc with various arguments */
#include <stdlib.h>
int main()
{
  int *p, *q, *r;
  int a, n = 10;
  void *s;
  p = (int *) malloc(2*sizeof(int));
  q = (int *) malloc(2*sizeof(a));
  r = (int *) malloc(sizeof(int));
  s = malloc(15);
  p = (int *) malloc(n*sizeof(int));
  return(0);
}
