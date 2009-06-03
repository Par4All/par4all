// check use of constant increment and loop bound by semantics analysis

//#include <stdio.h>

int for03()
{
  int x[512];
  int i;
  int j;

  for(i = 0, j = 0; i<500;i +=128, j++)
    x[i] = 0;

  //printf("%d, %d\n", i, j);

 return 0;
}

main()\
{
  for03();
}
