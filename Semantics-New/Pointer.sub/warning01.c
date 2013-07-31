// warning, we can to access a var not initialize

#include<stdlib.h>

int main()
{
  int i;
  int *p;
  
  if (rand())
    p = malloc(sizeof(*p));
  
  i = *p;
  
  return 0;
}
