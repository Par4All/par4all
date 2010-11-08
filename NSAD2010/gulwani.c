//    Test from Gulwani 2007
//

#include <stdio.h>

int main()
{

  int x,y,z;

  x=0;
  y=50;

  while(x<100) {
    if (x<50) 
      x++;
    else  {
      x++; y++;
    }
  }

  if (y==100) printf("property  verified\n");
 
}
