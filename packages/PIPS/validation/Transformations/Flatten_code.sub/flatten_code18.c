/* Check that property opBasic test case: the second "i" declaration ("int i = 2") conflicts
   with the first one, it will need to be rewritten.

   Also the initialization of the second i is constant, an
   initialization statement must be added because it is located in a
   control flow cycle, namely a for loop.

   Finaly, the loop is unrollable.
 */

#include <stdio.h>

void foo()
{
}

int flatten_code18()
{
  int k;
  float a[3];

  for (k=0; k<3; k++)
  {
    a[k] = 0.;
    foo();
  }

  for (k=0; k<3; k++)
  {
    a[k] = 0.;
  }

  return k;
}

int main(int argc, char **argv)
{
  flatten_code18();
}
