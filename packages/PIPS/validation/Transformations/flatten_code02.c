/* In this test case, we have two declaration conflicts, one
   concerning "i" and one concerning "k".

   This test case differs from test case "flatten_code_01" in that the
   conflict between the "k" declarations doesn't occur in the
   top-level block itself.

   Initializations are constant expressions. No initialization
   statements are added.

   Note: this test case could be used in Semantics as well.
 */

#include <stdio.h>

int flatten_code02()
{
  int i = 1;
  int j;
  i++;
  {
    int i = 2;
    int k = 2;
    i++;
    j = k;
    {
      int k = 3;
      i++;
      j = k;
    }
  }
  i=j;

  printf("%d %d\n", i, j);
}

int main(int argc, char **argv)
{
  flatten_code02();
}



