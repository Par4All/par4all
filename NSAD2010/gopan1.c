//  From Gopan 2006
//
#include <stdio.h>

#define true (1)

int main()
{

  int x,y;

  x=y=0;

  while(true) {
    if (x<=50)
      y++;
    else y--;
    if (y<0) break;
    x++;
  }

  if(x==102)
    printf("property  verified\n");
}
