#include <stdlib.h>
#include <stdio.h>

int alea(void)
{
  float fr = ((float) rand())/((float)RAND_MAX);
  return ((fr>0.5)?1:0);
}


int main()
{

  int s = 0, t = 0, d = 0;

  while(s <= 2 && t <= 3) {
    if(alea())
      t++, s = 0;
    else
      d++, s++;
  }


  if(d <= 10)
    printf("healthy");
  else
    printf("crashed!");
}
