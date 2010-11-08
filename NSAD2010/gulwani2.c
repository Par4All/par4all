//   From  Gulwani 2007
//  Cited in Control-flow refinement and  Progress invariants for Bound
//  Analysis - PLDI'09

#include <stdio.h>

int main()
{

  int x,y,z;

  x=0;
  y=50;

  while(x<100)
    {
      while ( x<50)
	x++;
      while (x<100 && x>=50){
	x++; y++;
      }
    }
 
  if (x ==100 && y==100) printf("property  verified\n");
}
