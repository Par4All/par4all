// Julien Henry (Verimag)
// Path focusing...
// GDR GPL 2011, Lille 2011-06-08, slide 8
// loop with two computation phases
// to test path dependent analyses

#include <stdio.h>
#define true 1

int main(void)
{
  int x = 0;
  int y = 0;
  while (true)
  {
    fprintf(stdout, "loop: x=%d y=%d\n", x, y);
    if (x<=50)
      y++;
    else
      y--;
    if (y<0) break;
    x++;
  }
  fprintf(stdout, "end: x=%d y=%d\n", x, y);
  return 0;
}
