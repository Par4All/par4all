// error, we want to access a var not initialize

#include<stdlib.h>

int main()
{
  int i;
  int *p;
  
  i = *p;
  
  return 0;
}
