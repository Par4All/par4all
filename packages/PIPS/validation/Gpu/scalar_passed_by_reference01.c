#include <stdlib.h>
#include <math.h>



int main() {
  int j1, j2, i,j; // loop indices
  int n = rand();
  int m = rand();

  double float_n = 321414134.01;
  double symmat[m][m];
  double data[n][m];
  double mean[m];

  // sum doesn't have to be copied out, thus do not pass it by reference !
  for (j = 1; j <= m; j++) {
    double sum = 0;
    for (i = 1; i <= n; i++)
      sum += data[i][j];
    mean[j] = sum / float_n;
  }
   
  // Force a region out for mean
  double x;
  for (j = 1; j <= m; j++) {
    x+= mean[j];
  }


  return 0;
}
