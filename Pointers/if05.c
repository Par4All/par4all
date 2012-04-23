// Impact of conditions on points-to

// Check the and operator

#include <stdio.h>

int main() {
  int *p=NULL, *q=NULL, i, j;

  if(!p&& !q) {
    p = &i;
    q = &j;
  }
  else if(!p) {
    *q = 2;
  }
  else if(!q) {
    *p = 1;
  }
  else {
    *p = 1;
    *q = 2;
  }
  
  return 0;
}
