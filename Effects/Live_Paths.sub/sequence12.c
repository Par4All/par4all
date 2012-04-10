#include <stdio.h>

int main() {
  int a=0;
  
  int *p;
  
  // We don't know where p is pointing, thus a may be lived
  printf("%d",*p);

}
