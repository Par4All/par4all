#include <stdio.h>

int main() {
  int i;
  int n = 100;
  float a[n];
  float tmp1 = 0.0;
  float tmp2 = 0.0;

  //salgadou
#pragma I Like
  for(i = 0; i < n; i++)
    a[i] = i+100;

  //la metchikabou
#pragma To move it
  for(i = 0; i < n; i++) {
    tmp1 += a[i]*10;
    tmp2 += a[i]*5;
  }

  printf("%f %f %f %f", tmp1, tmp2);
  for(i = 0; i < n; i++)
    printf("%f ", a[i]);

  return 0;
}
