/* Impact of conditions on points-to
 *
 * Check equality to NULL
 */

#include <stdio.h>

int main() {
  int *p, i, j;

  p = &i;

  if(p==NULL)
    p = &i;
  else
    p = &j;
  
  return 0;
}
