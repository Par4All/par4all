/* Check that successive initializations are properly taken into
   account. */
#include <stdio.h>
void static01()
{
  static int i = 0;

  i++;
  printf("%d\n", i);
}

main()
{
  static01();
  static01();
  static01();
}
