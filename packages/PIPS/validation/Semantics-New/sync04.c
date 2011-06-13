// N. Halbwachs 2010-12-09 Aussois slide 43

// Multiple loop example, apparently taken from Laure Gonnord's PhD

// http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/pdfs/HALBWACHS.Nicolas.aussois2010.pdf

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

bool alea(void)
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

    // The expected invariant is given as a 3-D figure in d, s and t
    // s<=4, s<=d, ???
    // found by PIPS: {s<=d, d<=s+4t, 0<=s, s<=4}
    fprintf(stdout, "t=%d d=%d s=%d\n", t, d, s);
  }
}
