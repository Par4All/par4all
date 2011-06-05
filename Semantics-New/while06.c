// FI: Just a check for loops with a non-convex while condition

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

bool alea(void)
{
  return rand()%2;
}

void while06()
{
  int x=0, y=0;

  while(x<=4 || y<=8) {
    if(alea()) x++;
    if(alea()) y++;
  }
  fprintf(stdout, "x=%d y=%d\n", x, y);
}

main()
{
  while06();
}
