/* FI: check double renaming */

#define xe x
#define xi1 x
#define xi2 x

#include <stdio.h>

static int block_scope13()
{
  int xe = 6;
  {
    int xi1 = 7;
  lab1: printf("First internal x=%d\n",xi1);
    xi1++;
  }
  {
    static int xi2 = -1;
    xi2++;
    printf("Second internal x=%d\n", xi2);
    if(xi2<=0) goto lab1;
  }
  printf("External x=%d\n", xe);
  return xe;
}

main()
{
  int i;
  i = block_scope13();
  return 0;
}
