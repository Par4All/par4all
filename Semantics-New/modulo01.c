/* Bug: side effects ignored in modulo based expressions. see
   modulo_to_transformer(). */

#include <stdio.h>

int main()
{
  int i=4;
  int j, k;

  j = (i++)%2;
  k = (++i)%2;

  printf("i=%d, j=%d, k=%d\n", i, j, k);

  return 0;
}
