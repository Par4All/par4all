// see Semantics-New/Pointer.sub/if07 to the same test with pointers

#include<stdlib.h>

int main()
{
  int i, j, k, l;
  
  if (k<=i && i<=l) {
    if (rand()) {
      i=0;
    } else {
      j=0;
    }
    rand();
  }
  
  return 0;
}
