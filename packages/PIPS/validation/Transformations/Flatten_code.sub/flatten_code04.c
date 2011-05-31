/* Basic test case: the second "i" declaration ("int i = 2") conflicts
   with the first one, it will need to be rewritten.

   Note: constant initializations are used;
 */

#include <stdio.h>

int flatten_code04()
{
  int i = 1;
  int j;
  i++;
  if (1)
  {
    int i = 2;
    i++;
    j += i;
    {
      int i = 3;
      i++;
      j += i;
    }
  }
  i = j;

  printf("%d %d\n", i, j);
}

int main(int argc, char **argv)
{
  flatten_code04();
}
