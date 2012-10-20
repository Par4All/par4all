/* To check behavior with non returning functions, such as error
   handling functions.
 */

#include <stdlib.h>

void call29(int * q)
{
  *q = 1;
  exit(1);
  *q = 2;
  return;
}

int call29_caller(int * qq)
{
  int i = 0;
  call29(qq);
  return i++; // To check indirect impact on memory effects
}
