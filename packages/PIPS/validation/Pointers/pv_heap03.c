/* basic free */
#include <stdlib.h>
int main()
{
  int *p;
  int a;
  p = (int *) malloc(2*sizeof(int));
  p[0] = 0;
  p[1] = 1;
  free(p);
  return(0);
}
