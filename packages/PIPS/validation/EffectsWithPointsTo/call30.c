/* To check behavior with a free
 */

#include <stdlib.h>

void call30(int * q)
{
  free(q);
  return;
}

int call30_caller(int * qq)
{
  int i = 0;
  call30(qq);
  return i++; // To check indirect impact on memory effects
}
