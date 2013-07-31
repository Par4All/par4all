// same that malloc02 but we free on p and not on *pp
// want p=*HEAP*_...
// improvement, semantic when we free pp or p?? we lost p=*HEAP*_...

#include<stdlib.h>

int main()
{
  int *p;
  int **pp;
  
  pp = malloc(sizeof(*pp));
  *pp = malloc(sizeof(**pp));
  p = *pp;
  
  free(p);
  free(pp);
  
  return 0;
}
