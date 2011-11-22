// FI: Just a check for loops with a non-convex exit condition

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

bool alea(void)
{
  return rand()%2;
}

void while05()
{
  int x=0, y=0;

  while(x<=4 && y<=8) {
    if(alea()) x++;
    if(alea()) y++;
  }
  // PIPS: {x<=5, 45<=9x+5y, x+y<=12, y<=9}
  fprintf(stdout, "x=%d y=%d\n", x, y);
}

main()
{
  while05();
}
