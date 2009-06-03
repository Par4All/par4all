// check use of constant increment and loop bound by semantics analysis

//#include <stdio.h>

void comma03()
{
  int i;
  int j;
  int k;
  float x;

  i = (j = 1, k = 2, x = 3.5);

  //printf("%d %d %d\n", i, j, k);
  i = 0;
}

main()
{
  comma03();
}
