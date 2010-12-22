// N. Halbwachs 2010-12-09 Aussois slide 40
// Exemple Laure Gonnord 2007
//
// http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/pdfs/HALBWACHS.Nicolas.aussois2010.pdf

// Le graphe de controle a ete construit ici directement par Fabien
// Coelho, sans utiliser les techniques de Vivien Maisonneuve

#include <stdio.h>
#include <stdlib.h>
// Just for the true in the while loop condition...
#include <stdbool.h>

bool alea(void)
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
    // Which invariant do you expect? Just a figure in the slide
    // something like v>=(t-9)/5, v<=t, v <=t/2+3 (numerical
    // coefficients are guessed); variable x and d must be projected
    fprintf(stdout, "v=%d t=%d x=%d d=%d\n", v, t, x, d);
  }
}
