// error, we want to access a var initialize to NULL

#include<stdlib.h>

int main()
{
  int i, j=0;
  int *p, *q;
  p = NULL;
  
  q = p+(1+j);
  i = q-p;
  
  return 0;
}
