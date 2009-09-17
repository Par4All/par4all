/* To make sure that the comma expressions are not surrounded by
   useless parentheses */

#include <stdio.h>

int foo(int i)
{
  /* Default type is int */
  static l;
  l = i;
  return i;
}

int main()
{
  int i,j,n,BlockEnd=3;

  for(i=1;i<16; i<<=2) {
    for ( j=i, n=0; n < BlockEnd; j++, n++ ) {
      int k, l;
      k = foo((j,n));
      l = k;
      printf("%d\n", l);
    }
  }
}
