// simple case

#include<stdlib.h>

int main()
{
  int *p;
  
  p = malloc(sizeof(*p));
  free(p);
  
  return 0;
}
