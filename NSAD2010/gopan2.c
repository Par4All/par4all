//  Test from Gopan - SAS07
//  Cited by Gulwani in Control-flow refinement and  Progress invariants for Bound
//  Analysis - PLDI'09
// 

#include <stdio.h>

int main()
{

  int x,y,z,k;

  x=0;
  y=0;

  while (y>=0) 
    {
     while(y>=0 && x<=50)
	{
	  x++; y++;
	}
      while (y>=0 && x>50)
	{
	  y--;
	  x++;
	}
    }
  if(x==103)
    printf("property  verified\n");
}
