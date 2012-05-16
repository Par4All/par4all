/* Check that non-freeable locations are not freed.
 */

#include <malloc.h>

int main()
{
  int i=1, *p=&i;

  free(p);

  return 0;
}
