#include <stdio.h>
#include <stdlib.h>

int alea(void)
{
  float fr = ((float) rand())/((float)RAND_MAX);
  return ((fr>0.5)?1:0);
}


int main()
{
  float z;
  int x,l,t;

  t=l=x=0;
  while(1)
    {
      x=0;
      while (x<=9 && alea())
	{
	  x++; t++;l++;
	}
      x=0;
      while(x<=49 || alea())
	{
	  x++;t++;
	}
      if (6*l<= t+5*x) printf("property verified\n");
    }
}
