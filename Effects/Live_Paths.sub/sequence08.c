// test liveness analysis on pointer;
#include <stdio.h>

int main () {
  int a[5];
  int i;
  int* p;

  p = NULL;

  for (i = 0; i < 5; i++) {
    a[i] = i;
  }

  p = &(a[2]);

  return *p;
}
