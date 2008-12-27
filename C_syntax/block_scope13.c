#include <stdio.h>

static int block_scope13()
{
  int x = 6;
  {
    int x = 7;
  lab1: fprintf(stdout, "%d\n",x);
  //x++;
  }
  {
    //int x = -1;
    goto lab1;
  }
  return x;
}

main()
{
  block_scope13();
}
