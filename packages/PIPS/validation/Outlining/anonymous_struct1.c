#include <stdio.h>

int main() {
  int i, j, k;
  struct { double x[17][13][11]; double y[17][13][11]; } a;
kernel:
  for(i = 0; i < 17; i++)
    for(j = 0; j < 13; j++)
      for(k = 0; k < 11; k++) {
        a.x[i][j][k] = 3*i+100+5*j-7*k;
        a.y[i][j][k] = 3*i-100+5*j-7*k;
      }

  for(i = 0; i < 17; i++)
    for(j = 0; j < 13; j++)
      for(k = 0; k < 11; k++)
        printf("%f %f ", a.x[i][j][k], a.y[i][j][k]);
  puts("");

  return 0;
}
