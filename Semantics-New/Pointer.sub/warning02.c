// warning, we can access a var initialize to NULL

#include<stdlib.h>

int main()
{
  int i;
  int *p;
  p = NULL;
  
  if (rand())
    p = malloc(sizeof(*p));
  
  i = *p;
  
  return 0;
}
