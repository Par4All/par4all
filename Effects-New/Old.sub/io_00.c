#include <stdio.h>

void test(FILE *fp)
{
  fpos_t pos;
  (void) fgetpos(fp, &pos);
}

