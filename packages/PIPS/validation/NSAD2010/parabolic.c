// Example in Section 3.4

#include <stdio.h>


int main()
{
  int i = 0, j = 0, n;
  if(n<0) exit(1);
  while(i<=n) {
    i++;
    j+=i;
  }
  printf("i=%d, j=%d\n", i, j);
}
