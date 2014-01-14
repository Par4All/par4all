/* Check that non-freeable locations are not freed.
 */

#include <malloc.h>

int main()
{
  int *p = (int *) malloc(10*sizeof(int));

  p++;

  free(p);

  return 0;
}
