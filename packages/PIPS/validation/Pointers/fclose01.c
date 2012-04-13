/* For intrinsics fclose() and fopen() */

#include <stdio.h>

void fclose01()
{
  FILE * f;
  f = fopen("toto", "r");
  fclose(f);
  return;
}
