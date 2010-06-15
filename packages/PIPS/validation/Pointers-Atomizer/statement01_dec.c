#include <stdio.h>
int main() {
  int j=0,k, tab[5]; 
  int *p = &j;
  int *q = tab[1];
  
  k=1;
  j=k; 
  p=&k;

  return 0;
}
