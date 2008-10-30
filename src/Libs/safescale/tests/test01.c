#include<stdio.h>

int main(int argc, char **argv)
{
  int i;
  int m = 0;
  int n = 5;

  /* BEGIN_KAAPI_traou */
  for (i = 1; i < n; i++)
  {
    m = m + n + i;
  }
  /* END_KAAPI_traou */

  printf("Result: %d\n", m);
  return 0;
}
