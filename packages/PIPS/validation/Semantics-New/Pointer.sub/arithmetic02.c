#include <stdlib.h>

int main()
{
  int i, j, *p, *q;
  
  if (rand()) {
    p = &i;
    q = &j;
  } else {
    p = &j;
    q = &i;
  }
  
  return 0;
}
