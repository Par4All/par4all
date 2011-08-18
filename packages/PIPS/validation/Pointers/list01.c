#include "list.c"

#include <stdio.h>

int main(void)
{
  list l = nil;
  int n;
  fscanf(stdin, "%d", &n);
  if (n>0)
  {
    while (n--)
    {
      l = list_cons((double) n, l);
    }
  }
  n = list_len(l);
  fprintf(stdout, "n=%d", n);
  return 0;
}
