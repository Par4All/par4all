// N. Halbwachs 2010-12-09 Aussois

#include <stdio.h>
#include <stdlib.h>

int alea(void)
{
  // rand() >=0
  // 0 <= return <= 1
  return rand()%2;
}

int main(void)
{
  int x=0, y=0;
  while (x<=100)
  {
    if (alea())
      x=x+2;
    else
      x=x+1,y=y+1;
  }
  // immediate widening: :0 <= y <= x, x >= 101
  // iterated once: 0 <= y <= x, 101 <= x <= 102, x+y<=202
  fprintf(stdout, "x=%d y=%d\n", x, y);
}
