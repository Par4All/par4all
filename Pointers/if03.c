// Impact of conditions on points-to

// Too bad this code is dead wrong

#include <stdio.h>

int main() {
  int *p=NULL, i;

  if(p) {
    p = &i;
  }
  else {
    *p = 1;
  }
  
  return 0;
}
