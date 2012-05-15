/* Bug: free should not work when a numerical non-zero offset appears*/

#include<stdlib.h>

int main()
{
  int *p, *q, *r;
  p = malloc(10*sizeof(int));
  p++;
  q = p-1;
  r = p;
  free(p);
  return 0;
}
