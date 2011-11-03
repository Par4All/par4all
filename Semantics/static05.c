#include <stdio.h>
void static05()
{
  static int i;
  // dangerous but possible
  printf("%d\n", i);
}

main()
{
  static05();
  static05();
  static05();
}
