/* More state variables */

#include <stdio.h>
#include <stdlib.h>

int alea(void)
{
  return rand()%1;
}


int main()
{
  float z;
  int u,l,t,v;

  u=l=t=v=0;
  while(1) {
    v = 0;
    for(u=0; u<60;u++) {
      if(v<=9 && alea())
	l++, v++;
      t++;
      if (6*l<= t+50) printf("property verified\n");
    }
  }
}
