/* Check that possibly freeable locations are freed.
 *
 * Intraprocedural test of main
 */

#include <malloc.h>

int * foo(void)
{
  int *p = (int *) malloc(sizeof(int));
  return p;
}

int main()
{
  int i=1, *p=foo();

  free(p);

  return 0;
}
