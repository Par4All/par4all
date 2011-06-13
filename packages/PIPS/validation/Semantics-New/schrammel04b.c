// From Schrammel and Jeannet, NSAD 2010, p. 85, Example 5.1

// Doubly nested, outer loop on p

// Simplified version of schrammel04.c: additional tests removed

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int alea(void)
{
  return rand() %2;
}

void main()
{
  int x1=foo(),y,x2=2-foo(),p=1;
  y = foo();
  while(p<=20) {
    //
    // Expected result by Schrammel/Jeannet (the third inequality is also the
    // second one):
    // 1<=x1+x2, 0<=x2<=1, 0<=x1
    // In fact, 0 <= x2 <=2 because of the first iteration

    // Also true: 0<=x1<= ~p/2
    //printf("p=%d, x1=%d, x2=%d\n", p, x1, x2);
    while(2*x1+2*x2<=p && alea()) {
      x1 = x1 + y + 1;
      x2 = y;
      y = foo();
    }
    p++;
  }

  y = y;
}

int foo()
{
  return rand() %2;
}
