// From Schrammel and Jeannet, NSAD 2010, p. 80, Exmaple 4.5

#include <assert.h>

void main()
{
  int x1,y,x2;
  assert(0<=x1 && x1<=x2 && x2<=1);
  y = foo();
  while(x1+x2<=4 && 1<= y && y <= 2) {
    x1 = x1 + 2*y - 1;
    x2 = x2 + y;
    y = foo();
  }
  // Expected result: 0<=x1, x2<=x1+1, x1+x2<=9, 4x2<=2x1+9, 2x1<=3x2
  y = y;
}

int foo()
{
  int x = 3;
  return x*x*x/54;
}
