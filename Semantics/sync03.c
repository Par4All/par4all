// N. Halbwachs 2010-12-09 Aussois slide 40
// Exemple Laure Gonnord 2007

#include <stdio.h>
#include <stdlib.h>

typedef enum { false, true } boolean;

boolean alea(void)
{
  return rand()%2;
}

int main(void)
{
  int v, t, x, d;
  v = t = x = d = 0;

  while (true)
  {
    if (alea() && x<=4)
      x++, v++;
    if (alea() && d<=9)
      d++, t++;
    if (alea() && d==10 && x>=2)
      x=0, d=0;
    fprintf(stdout, "v=%d t=%d x=%d d=%d\n", v, t, x, d);
  }
}
