//   from  Gulwani Test 2006
//  Control-flow refinement and  Progress invariants for Bound
//  Analysis - PLDI'09

#include <stdio.h>



int main()
{

  int x,y,z,lock;

  x=1;
  lock = 0;
  y=0;


  while(x!=y)
    {
      scanf("%d",&z);
      if (z>=0) {
	lock =1; x=y;
	lock =0; y++;
      }
      else {
	lock =1; x=y;
      }
  }


  if (lock ==1) printf("property  verified\n");

}
