// Periodic example: "unrolling" of periodic4a.c by a factor of two

// It works because T^2 is computed

// Case flip-flop 2, like periodic4c.c, but with the useless stuff
// about y and z

#include <stdio.h>


int main()
{

  int x;
  int new,old;
  int y,z,k;
  x=0;
  y=0;
  z=0;
  new=0;
  old=1;

  while (x<10)
    {
      if (new==0) y++;
      else z++;
      new = 1 - new;
      old = 1 - old;
      x++;
    }
  if((new==1 && old==0) || (new==0 && old==1))
    printf("property  verified\n");
  else
    printf("property not found\n");
}
