/* Example used in 
 *
 * Inductive Invariant Generation via Abductive Inference
 *
 * Isil Dilig, Thomas Dilig, Boyang Li, Ken McMillan
 *
 * OOPSLA 2013, pp. 443-456
 *
 * FI: I add a main function for the cloning
 */

#include <assert.h>

void foo(int flag)
{
  int i, j = 1, a = 0, b = 0;
  float x;
  if(flag) i = 0;
  else i = 1;
  while(x>0.) {
    a++;
    b += (j-i);
    i += 2;
    if(i%2==0) j += 2;
    else j++;
  }
  if(flag) assert(a==b);
}

int main()
{
  foo(0);
  foo(1);
  return 0;
}
