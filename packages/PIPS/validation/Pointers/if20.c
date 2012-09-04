// Impact of conditions on points-to

// Check the use of pointer comparisons to remove useless arcs

// Opposite of if08.c

#include <stdio.h>

int main() {
  int *p, *q, *r, i, j;

  p = &i;

  if(i>0) q = &i;

  if(p!=q) {
    r = q;
  }
  else {
    r = &i;
  }
  
  return 0;
}
