//  car-safety + Symbolic bound for t

#include <stdio.h>
#include <assert.h>
#include <stdlib.h>

int alea(void)
{
  float fr = ((float) rand())/((float)RAND_MAX);
  return ((fr>0.5)?1:0);
}

void main(int n)
{
  int s = 0, t = 0, d = 0;


  assert(n>=0);
  while(s <= 2 && t <= n) {
    if(alea())
      t++, s = 0;
    else
      d++, s++;
  }

  if(d <= 2*n+3)
    printf("healthy");
  else
    printf("crashed!");
}
