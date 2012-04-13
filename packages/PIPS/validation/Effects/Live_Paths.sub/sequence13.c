#include <stdio.h>

int main() {
  int a=0;
  
  int *p, *q;

  p = &a;
  q = p;
  
  printf("%d",*q);

}
