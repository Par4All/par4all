/* The initialization of q is wrong. The type mismatched should be
 * detected before an empty list of cells is returned for tab[1].
 */

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
