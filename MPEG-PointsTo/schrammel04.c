// From Schrammel and Jeannet, NSAD 2010, p. 85, Example 5.1

// Doubly nested, outer loop on p

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

int alea(void)
{
  return rand() % 2;
}

void main()
{
  int x1=foo(),y,x2=2-foo(),p=1;
  // With a proper foo() returning a value between 0 and 1, the
  // following assert is useless
  // assert(0<=x1 && 1<=x2 && x1+x2<=2 && p==1);
  y = foo();
  while(p<=20) {
    printf("p=%d, x1=%d, x2=%d\n", p, x1, x2);
    if(2*x1+2*x2<=p) {
      if(1) {
	// The loop is not entered for small values of p because 2x1+2x2
	// >= 2 and p starts with 1, but it is certainly entered for p>=
	// 4, unless x1 has increased to fast during the previous
	// iterations
	//
	// So the internal loop is not entered till p has increased
	// enough. The loop is entered or not entered and a convex hull
	// with the identity transition must be performed
	//
	// Also, we may skip the loop at any time because of y
	//
	// The information published by Bertrand
	float z = 0.;

	while(2*x1+2*x2<=p && alea()) {
	  if(1) {
	    x1 = x1 + y + 1;
	    x2 = y;
	    y = foo();
	  }
	}
	// To get the loop postcondition
	//y = y;
	// without generating an indentity transformer
	z = z;
      }
      else {
	// do nothing at all, not even read y? Read y on every transition
	y = foo();
      }
    }
    p++;
  }
  //
  // Expected result by Schrammel (the third inequality is also the
  // second one):
  // 1<=x1+x2, 0<=x2<=1, 0<=x1

  // Also true: 0<=x1<=2

  y = y;
}

int foo()
{
  return alea();
}
