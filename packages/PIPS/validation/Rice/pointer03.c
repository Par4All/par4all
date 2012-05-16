/* Parallelizarion with pointers
 *
 * Same as pointer02, but with no offset
 */

#include <assert.h>

void pointer03(int n, float *p)
{
  assert(p!=0);
  int i;
  float * r = p;

  for(i=0; i<n; i++)
    p[i] = r[i];
}
