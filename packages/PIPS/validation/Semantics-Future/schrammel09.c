// From Schrammel and Jeannet, NSAD 2010, p. 85, Example 5.1

// Doubly nested, outer loop on p

// Simplified version of schrammel04.c: additional tests removed,
// assignment to x1 in inner loop removed
// preconditions on y not found, and then on x2.
// Linked to Bertrand Jeannet's question about nested loops.

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
    while(2*x1+2*x2<=p && alea()) {
      //x1 = x1 + y + 1;
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
