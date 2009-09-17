/* To make sure that the comma expressions are not surrounded by
   useless parentheses */

#include <stdio.h>

int foo(int i)
{
  return i;
}

int main()
{
  int j, n, BlockEnd=3;

  for ( j=1, n=0; n < BlockEnd; j++, n++ ) {
    int k, l;
    k = foo((j,n)), l = 1;
    printf("%d\n", k);
  }
}

