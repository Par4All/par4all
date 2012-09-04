/* Impact of conditions on points-to
 *
 * Check equality to NULL:  q cannot be NULL on return
 */

#include <stdio.h>

int if15(int *p) {
  int * q, i;

  if(p==NULL)
    q = &i;
  else
    q = p;
  
  return 0;
}
