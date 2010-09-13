// From Schrammel and Jeannet, NSAD 2010, p. 82, Example 4.9

#include <assert.h>

void main()
{
  int x1,y,x2;
  assert(0<=x1 && 1<=x2 && x1+x2<=2);
  y = foo();
  while(2*x1+2*x2<=7 && 0<= y && y <= 1) {
    x1 = x1 + y + 1;
    x2 = y;
    y = foo();
  }
  // Expected result: 1<=x1+x2, 0<=x2, 2x1<=2x2+9, 2x1+11x2<=22, 0<=x1
  y = y;
}

int foo()
{
  int x = 3;
  return x*x*x/54;
}
