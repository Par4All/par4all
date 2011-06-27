#include <stdio.h>

int main() {
  int i;
  int n = 100;
  float a[n];
  float tmp1 = 0.0;

  for(i = 0; i < n; i++)
    a[i] = i+100;

  for(i = 0; i < n; i++) {
    tmp1 += a[i]*10;
    tmp1 *= a[i]*5;
  }

  printf("%f %f %f %f", tmp1);
  for(i = 0; i < n; i++)
    printf("%f ", a[i]);

  return 0;
}
