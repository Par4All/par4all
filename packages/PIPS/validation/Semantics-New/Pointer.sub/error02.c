// error, we want to access a var initialize to NULL

#include<stdlib.h>

int main()
{
  int i;
  int *p;
  p = NULL;
  
  i = *p;
  
  return 0;
}
