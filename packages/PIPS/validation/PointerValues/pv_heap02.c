/* calloc with various arguments */
#include <stdlib.h>
int main()
{
  int *p, *q, *r;
  int a;
  void *s;
  p = (int *) calloc(2,sizeof(int));
  q = (int *) calloc(2,sizeof(a));
  r = (int *) calloc(1,sizeof(int));
  s = calloc(1,15);
  return(0);
}
