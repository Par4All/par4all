// Impact of conditions on points-to

// Check the use of undefined pointer comparisons

#include <stdio.h>

int main() {
  int *p, *q, *r, i, j;

  if(p==q) {
    r = &i;
  }
  else {
    r = &j;
  }
  
  return 0;
}
