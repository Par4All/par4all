/* Basic test case: the second "i" declaration ("int i = 2") conflicts
   with the first one, it will need to be rewritten.

   Its initialization is constant: it can be left in the new
   declaration. Not statememnt "i = 2;" is inserted.
 */

#include <stdio.h>

int split_initializations01()
{
  int i = 1;
  int j;
  i++;
  {
    int i = 2;
    i++;
    j=1;
  }
  i=j;

  printf("%d %d\n", i, j);
}

int main(int argc, char **argv)
{
  split_initializations01();
}
