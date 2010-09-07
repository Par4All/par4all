/* Model Checking Sequential Software Programs via mixed symbolic
   analysis
   Z. Yang & al.
   ACM Trans. on Design Automation of Electronic Systems
   26 pages, article 10, 2009, v. 14, n. 1

   Check the example in the paper above. foo() is never exited under
   the present preconditions after inlining.
*/


#include <stdio.h>

main()
{
  bar();
}

int bar() {
  int x = 3;
  int y = x - 3;

  while(x<=4) {
    y++;
    x = foo(x);
    // Execution: (x,y) = (4,1) and then (5,2) and the loop is exited
    printf("In loop: x=%d, y=%d\n", x, y);
  }
  y = foo(y);

  // Execution: (x,y) = (5,3)
  printf("At the end: x=%d, y=%d\n", x, y);
  return y;
}

/* In the context used, this functions returns s+1 */
int foo(int s)
{
  int t = s + 2;

  if(t>6)
    t -= 3;
  else
    t--;

  return t;
}
