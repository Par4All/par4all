/* free of a pointer which may have different heap targets */
#include <stdlib.h>
int main()
{
  int *p;
  int a;
  if (a== 1)
    p = (int *) malloc(2*sizeof(int));
  else
    p = (int *) malloc(3*sizeof(int));
  p[0] = 0;
  p[1] = 1;
  free(p);
  return(0);
}
