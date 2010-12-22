// N. Halbwachs 2010-12-09 Aussois, slide 23-24

#include <stdio.h>
#include <stdbool.h>

#define assert(b) if (!(b)) exit(1);

int main(void)
{
  bool b0 = false, b1 = false, ok = true;
  int x=0, y=0;

  assert(!b0 && !b1 && ok && x==0 && y==0);
  while (true)
  {
    if (b0 == b1)
      x = x+1;
    if (b0 ^ b1) // hmmm, no xor analysis yet because it's a bitwise operator
      y = y+1;

    // flip-flap-flop
    b0 = !b1;
    b1 = b0;

    ok = ok && (x>=y);
    // assert(ok);
    // we want ok==1
    fprintf(stdout, "b0=%d b1=%d ok=%d x=%d y=%d\n", b0, b1, ok, x, y);
  }
}
