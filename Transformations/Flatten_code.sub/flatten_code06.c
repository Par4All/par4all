/* Basic test case: the second "i" declaration ("int i = 2") conflicts
   with the first one, it will need to be rewritten.

   Also the initialization of the second i is constant, an
   initialization statement must be added because it is located in a
   control flow cycle, namely a for loop.

   Finaly, the loop is unrollable.
 */

#include <stdio.h>

int flatten_code06()
{
  int i = 1;
  int j;
  int k;
  float a[3];
  i++;
  for (k=0; k<3; k++)
  {
    int i = 2;
    i++;
    j=1;
    a[k] = 0.;
  }
  i=j;

  printf("%d %d\n", i, j);
}

int main(int argc, char **argv)
{
  flatten_code06();
}
