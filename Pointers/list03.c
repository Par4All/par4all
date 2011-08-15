#include "list.c"

#include <stdio.h>

int main(void)
{
  list l = nil;
  int n;
  fscanf(stdin, "%d", &n);
  while (n--)
  {
    l = list_cons((double) n, l);
  }
  n = list_len(l);
  fprintf(stdout, "n=%d", n);
  list_clean_2(l);
  return 0;
}
