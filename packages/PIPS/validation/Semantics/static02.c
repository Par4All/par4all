/* Check that external initializations are properly taken into
   account. */
#include <stdio.h>

void static02()
{
  static int i = 0.;

  i++;
  printf("%d\n", i);
}

main()
{
  static02();
  static02();
  static02();
}
