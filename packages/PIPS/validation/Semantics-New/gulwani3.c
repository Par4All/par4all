//   from  Gulwani Test 2007
//  Control-flow refinement and  Progress invariants for Bound
//  Analysis - PLDI'09

#include <stdio.h>



int main()
{

  int x,y,z;

  x=1;
  y=0;
  z=0;

  while(x<100)
    {
      while ( x<=50)
	x++;
      while (x<=100 && x>50){
	x++; y++;
      }

      z=y;
    }


  if (x ==101 && y==50) printf("property  verified\n");

}
