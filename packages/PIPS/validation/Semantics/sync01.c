// N. Halbwachs 2010-12-09 Aussois, slide 16-18

#include <stdio.h>
#define assert(b) if (!(b)) exit(1);

//typedef enum { false, true } boolean;
#include <stdbool.h>

int main(void)
{
  //bool b = false, ok = true;
  //int x=0, y=0;
  bool b, ok;
  int x, y;

  assert(!b && ok && x==0 && y==0);
  while (true)
  {
    if (b)
      y = y+1;
    else
      x = x+1;
    // flip-flop
    b = !b;
    ok = ok && (x>=y);
    // If the previous boolean assignment is changed into
    //if(x<y)
    //  ok = false;
    // the condition is found.

    // assert(ok);
    // we want ok==1
    fprintf(stdout, "b=%d ok=%d x=%d y=%d\n", b, ok, x, y);
  }
}
