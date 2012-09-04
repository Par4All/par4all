/* Impact of conditions on points-to
 *
 * Check equality to NULL:  q cannot be NULL on return
 */

#include <stdio.h>

int if16(int *p) {
  int * q, i;

  if(p!=NULL)
    q = p;
  else
    q = &i;
  
  return 0;
}
