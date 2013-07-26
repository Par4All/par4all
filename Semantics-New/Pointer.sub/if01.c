#include<stdlib.h>


int main()
{
  int i, j, k, *p;
  i=0;
  j=1;
  
  if (rand()) {
    p = &i;
    k = i;
  } else {
    p = &j;
    k = j;
  }
  
  *p = 10;
  
  return 0;
}
