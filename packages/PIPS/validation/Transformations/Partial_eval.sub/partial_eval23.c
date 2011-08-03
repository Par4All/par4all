#include <stdlib.h>

int main(void)
{
  int i = 0;
  int j = rand();
  // identity
  j |= i;
  int k = rand();
  int l = k | i;
  int m = rand();
  // absorbtion
  m &= i;
  // should be zero
  return (j+l) & m;
}
