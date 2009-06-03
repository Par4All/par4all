// check use of constant increment and loop bound by semantics analysis

//#include <stdio.h>

void comma02()
{
  int i;
  int j;
  int k;

  i = (j = 1, k = 2);

  //printf("%d\n", i);
  i = 0;
}

main()
{
  comma02();
}
