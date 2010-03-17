/* Dealing with signed/unsigned conversions */

#include <stdio.h>

int main(void)
{
  int i;
  unsigned int ui;

  i = -1;
  ui = i;

  fprintf(stderr, "ui = %u\n", ui);
}
