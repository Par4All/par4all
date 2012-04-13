/* free of a target through a conditional operator */
#include <stdlib.h>
int main()
{
  int *p, *q;
  int a=1;
  p = (int *) malloc(2*sizeof(int));
  q = (int *) malloc(2*sizeof(int));
  free(a==0?p:q);
  return(0);
}
