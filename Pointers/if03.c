// Impact of conditions on points-to

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
