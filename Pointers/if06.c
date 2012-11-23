// Impact of conditions on points-to

// Check the impact of side effects

#include <stdio.h>

int main() {
  int a[10];
  int *p=&a[0];

  if(p++) {
    *p = 1;
  }
  else {
    *p = 0;
  }

  return 0;
}
