// N. Halbwachs 2010-12-09 Aussois slide 43

#include <stdio.h>
#include <stdlib.h>

typedef enum { false, true } boolean;

boolean alea(void)
{
  return rand()%2;
}

int main(void)
{
  int t = 0, d = 0, s = 0;
  while (true)
  {
    while (alea() && s<=3) s++, d++;
    while (alea()) t++, s=0;
    fprintf(stdout, "t=%d d=%d s=%d\n", t, d, s);
  }
}
