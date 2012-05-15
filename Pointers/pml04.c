#include<stdlib.h>
int main()
{
  int *p;
  p = malloc(10*sizeof(int));
  p++;
  free(p);
  return 0;
}
