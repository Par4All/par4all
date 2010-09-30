// From Schrammel and Jeannet, NSAD 2010, p. 83, Example 4.10

// Expected result:

#include <assert.h>

void main()
{
  int x1,y,x2;
  assert(x1<=1 && x2<=1 && x1+x2>=1);
  y = foo();
  while(2*x1+x2+y<=6 && x2-y <=2 && 0<=y && y <= 1)
    {
      x1 = x1+y+1;
      x2++;
      y = foo();
    }
  // Expected result: 1<=x1+x2, x1<=x2+2, x2<=x1+1, x1<=2x2+1, x2<=3, 2x1+x2<=10
  y = y;
}

int foo()
{
  int x = 3;
  return x*x*x/54;
}
