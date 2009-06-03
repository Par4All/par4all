// check use of constant increment and loop bound by semantics analysis

// #include <stdio.h>

void comma06()
{
  int i;
  int j;
  int k;
  int l;
  int m;

  i = (j = 2, k = 3, l = j*k*m*m);

  // printf("%d %d %d %d\n", i, j, k, l);
  i = 0;
}

main()
{
  comma06();
}
