// Check boolean comparisons

#include <stdbool.h>

int main(void)
{
  bool b1, b2;
  int i = 1, j = 2, k;

  b1 = i <= j;
  b2 = i>j;

  if(k)
    i = 2, j =1;

  b1 = i <= j;
  b2 = i>j;

  return 0;
}
