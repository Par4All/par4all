// STIC 2012, ASOPT presentation, Bertrand Jeannet: use zonotopes to
// find out that y is in [-0,25,0] after the test on x

// x and y are assumed integer numbers here, hence the invariant is y==0

// Modeling of non-linear computations

// Same as asopt01, but the non-linear expression is factorized

#include <stdio.h>
#include <assert.h>

void asopt02(int x, int y)
{
  int i = 0;
  assert(-1<=x && x<=1 && -1<=y && y<=1);
  if(x<=0) {
    // Expect y==0, obtain eventually y==0
    //
    // if e1 * e2 is analyzed and if e1 is bounded, e2 should be
    // analyzed again after adding e1==l or e1==u.
    //
    // Furthermore, once e1 is set to a constant, if e2 is affine then
    // the result is the convex hull of l*e2 and u*e2.
    //
    // Here if x==-1 then y==-x-1 and if x==0 then y==0. The convex
    // hull should yield y==0.
    //
    // More generally, we could look at the sign of the first and
    // second derivatives over the evaluation internal. This would not
    // work here, unless we look at the minimum of the function over
    // real numbers.
    y = x*(x+1);
    scanf("%d", &x); // To project x and get simple information about y
    i++; // to anchor the result
    i--; // to avoid involving i too much in the final precondition
    // and to look for a display bug...
  }
  return;
}
