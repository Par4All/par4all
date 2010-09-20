//   from  Gulwani Test 2006
//  Control-flow refinement and  Progress invariants for Bound
//  Analysis - PLDI'09

#include <stdio.h>

float alea(void)
{
  return 1.;
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
      z=alea();
      lock =1; x=y;
      if (z>=0.) {
	lock =0; y++;
      }
    }


  if (lock ==1) printf("property  verified\n");
 
}
