// Example in Section 3.4

#include <stdio.h>

int main()
{
  int i = 0, j = 2, k = 1;
  while(k<=10) {
    j--;
    i += j;
    k++;
  }

  printf("i=%d, j=%d, k=%d\n", i, j, k);
}
