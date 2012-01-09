// STIC 2012, expose ASOPT, Bertrand Jeannet

// Modelisation de calcul non-lineaire

#include <stdio.h>
#include <assert.h>

void asopt01(int x, int y)
{
  int i = 0;
  assert(-1<=x && x<=1 && -1<=y && y<=1);
  if(x<=0) {
    y = x*x+x;
    scanf("%d", &x);
    i++;
  }
  return;
}
