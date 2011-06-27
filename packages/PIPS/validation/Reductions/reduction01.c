#include <stdio.h>

int main() {
  int i;
  int n = 100;
  float a[n];
  float tmp1 = 0.0;
  float tmp2 = 0.0;
  float tmp3 = 0.0;
  float tmp4 = 0.0;

  for(i = 0; i < n; i++)
    a[i] = i+100;

  for(i = 0; i < n; i++) {
    tmp1 += a[i]*10;
    tmp2 += a[i]*5;
  }

  for(i = 0; i < n/2; i++) {
    tmp1 += a[i]*15;
    tmp3 += tmp1;
    tmp1 -= a[i]*25;
  }

  printf("%f %f %f %f", tmp1, tmp2, tmp3);
  for(i = 0; i < n; i++)
    printf("%f ", a[i]);

  return 0;
}
