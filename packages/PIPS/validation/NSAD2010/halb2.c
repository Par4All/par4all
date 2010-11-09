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
  int x,y;

  x=y=0;

  while(x<=100)
    {
      if (alea())
	x = x+4;
      else {
	x=x+2;
	y++;
      }
      if (x+2*y<=204) printf("property verified\n");
    }
}
