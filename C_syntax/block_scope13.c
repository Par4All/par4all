#include <stdio.h>

static int block_scope13()
{
  int x = 6;
  {
    int x = 7;
  lab1: fprintf(stderr, "%d\n",x);
  }
  {
    int x = 8;
    goto lab1;
  }
  return x;
}
