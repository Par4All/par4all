#include <stdio.h>



int main() {
  int n = 10;
  int i,j,k;
  double c[n][n];
  double sum;

  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {
        c[i][j] = i*n+j;
    }
  }


  // The two following loop nest produce the same "sum"
  // But the second one used to be (wrongly) scalarized 
  sum = 0.;
  for (i = 0; i <= n - 2; i++) {
    for (j = i + 1; j <= n - 1; j++) {
      for (k = i + 1; k <= j - 1; k++)
        sum += c[i][k];
    }
  }
  printf("sum is : %f\n",sum);

  sum = 0.;
  for (i = 0; i <= n - 2; i++) {
    for (j = i + 1; j <= n - 1; j++) {
      for (k = i + 1; k <= j - 1; k++)
        sum += c[i][k];
      c[i][j] = 0;
    }
  }
  printf("sum is : %f\n",sum);


}
