//  From  Gulwani 2006
//  Cited in Control-flow refinement and  Progress invariants for Bound
//  Analysis - PLDI'09

#include <stdlib.h>
#include <stdio.h>

int alea(void)
{
  float fr = ((float) rand())/((float)RAND_MAX);
  return ((fr>0.5)?1:0);
}


int main()
{
  float z;
  int x,y,lock;

  x=1;
  lock = 0;
  y=0;

  while(x!=y)
    {
      lock =1; x=y;
      if (alea()) {
	lock =0; y++;
      }
    }

  if (lock ==1) printf("property  verified\n");
}
