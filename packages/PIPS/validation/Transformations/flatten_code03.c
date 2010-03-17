/* The conflicting declarations are not nested, but placed side by side.

   Initializations are constant and not in a control cycle.
*/

#include <stdio.h>

int flatten_code03()
{
  int i = 1;
  int j;
  i++;
  {
    int i = 2;
    i++;
    j = 1;
  }
  {
    int i = 3;
    i++;
    j = 1;
  }
  i = j;

  printf("%d %d\n", i, j);
}

int main(int argc, char **argv)
{
  flatten_code03();
}
