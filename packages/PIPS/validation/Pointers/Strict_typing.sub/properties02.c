/* Check the impact of points-to properties on the management of null
 * pointers and implicit array pointers.
 *
 * Same as properties01.c, but the tpips script is different
 */

#include <assert.h>

void properties02(int n, int *p)
{
  int i;
  int *q = &i;

  assert(p!=0);

  *p = 19;

  *q = 1;

  return;
}
