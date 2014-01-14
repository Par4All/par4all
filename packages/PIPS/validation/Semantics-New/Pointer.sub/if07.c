// see Semantics-New/if04 to the same test without pointers

#include<stdlib.h>

int main()
{
  int i, j, k, l;
  int *p;
  
  if (k<=i && i<=l) {
    if (rand()) {
      p = &i;
    } else {
      p = &j;
    }
    *p=0;
    rand();
  }
  
  return 0;
}
