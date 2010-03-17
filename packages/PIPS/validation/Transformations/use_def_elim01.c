#include <stdio.h>

void use_def_elim01(int *x)
{
    /* check aliases */
    int *y = x;
    y[0] = 123;
}

int main()
{
  int i;
  use_def_elim01(&i);
  printf("i = %d\n", i);
  exit(i);
}
