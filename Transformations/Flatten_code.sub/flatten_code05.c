/* In this test case, we have two declaration conflicts, one
   concerning "i" and one concerning "k".

   This test case differs from test case "flatten_code_01" in that the
   conflict between the "k" declarations doesn't occur in the
   top-level block itself.

   Note: initialization of the internal i is not constant. An
   initialization statement must be added.
 */

#include <stdio.h>

int flatten_code05()
{
  int i = 1;
  int j;
  j++;
  {
    int i = j + 1;
    int k = 2;
    i++;
    j=1;
    {
      int k = 2;
      i++;
      j=1;
    }
  }
  i=j;

  printf("%d %d\n", i, j);
}

int main(int argc, char **argv)
{
  flatten_code05();
}



