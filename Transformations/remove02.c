#include <stdio.h>
int main () {
  float a[10];
  float b[10][10][10][10][10];
  int m = 0;
  int n = 0;
  int o = 2;

l0:  for (m = 0; m < 10; m++) {
    n = n+1;
    a[m] = n;
  }

l1:  for (m = 0; m < 10; m++) {
    // comments 4
    n = n+1;
    o = 2+o;
    o = 2+o;
    n = n+5;
    b[0][0][1][m][0] = n;
  }
     printf("%d\n",o);

  return 0;
}

