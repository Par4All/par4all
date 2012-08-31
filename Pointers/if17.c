// Impact of conditions on points-to

// Check the use of pointer comparisons

// Opposite of if08.c

#include <stdio.h>

int main() {
  int *p, *q, i;

  p = q = &i;

  if(p==q) {
    p = NULL;
  }
  
  return 0;
}
