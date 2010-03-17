/* make sure that the current precondition is properly used when
   computing transformers */

/* Problem: ++i is executed right away when it is needed, i++ seems to
   be delayed till the whole expression is evaluated, which does not
   let us combine transformers of subexpressions in the latter
   case. Any information from the standard? */

#include <stdio.h>

void foo(int i, int j, int k1, int k2)
{
  printf("i=%d, j=%d, k1=%d, k2=%d\n", i, j, k1, k2);
}

void update05()
{
  int i = 1;
  int j;
  int k1 = 0, k2 = 0;

  j = (i++) + (i++);
  // gcc: i=3, j=2, k1=0, k2=0
  foo(i, j, k1, k2);

  i = 1;
  j = (k1=i++) + (k2=i++);
  // gcc: i=3, j=2, k1=1, k2=1
  foo(i, j, k1, k2);

  i = 1;
  j = (k1=i++) + (i==1);
  // gcc: i=2, j=2, k1=1, k2=1
  foo(i, j, k1, k2);

  i = 1;
  j = (k1=++i) + (k2=++i);
  // gcc: i=3, j=5, k1=2, k2=3
  foo(i, j, k1, k2);
}

main()
{
  update05();
}
