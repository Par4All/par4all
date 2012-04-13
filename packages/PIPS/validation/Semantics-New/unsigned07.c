/* Dealing with signed/unsigned conversions */

/* Like unsigned06.c, but without initialization */

#include <stdio.h>

int main(void)
{
  unsigned char ui;
  ui = -1;
  printf("%d\n", (int) ui);
  return (int) ui;
}
