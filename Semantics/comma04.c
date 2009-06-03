// check use of constant increment and loop bound by semantics analysis

// #include <stdio.h>

void comma04()
{
  int i;
  int j;
  int k;
  int l;

  i = (j = 2, k = 3, l = j*k);

  // printf("%d %d %d %d\n", i, j, k, l);
  i = 0;
}

main()
{
  comma04();
}
