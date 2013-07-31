// want p=*HEAP*_...
// improvement, semantic when we free pp or *pp?? we lost p=*HEAP*_...

#include<stdlib.h>

int main()
{
  int *p;
  int **pp;
  
  pp = malloc(sizeof(*pp));
  *pp = malloc(sizeof(**pp));
  p = *pp;
  
  free(*pp);
  free(pp);
  
  return 0;
}
