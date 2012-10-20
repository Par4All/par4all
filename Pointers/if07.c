// Impact of conditions on points-to

// Check the use of NULL (same as if04.c)

// Same bug in the source code as in if03.c: a NULL pointer is dereferenced

#include <stdio.h>

int main() {
  int *p=NULL, i;

  if(p!=NULL) {
    p = &i;
  }
  else {
    *p = 1;
  }
  
  return 0;
}
