#include "list.c"

#include <stdio.h>

int main(void)
{
  list l = nil;
  int n, i=0;
  fscanf(stdin, "%d", &n);
  if (n>0)
  {
    while (n--)
    {
      l = list_cons((double) n, l);
      /* Result? */
      i++;
    }
  }
  n = list_len(l);
  fprintf(stdout, "n=%d, i=%d\n", n, i);
  return 0;
}
