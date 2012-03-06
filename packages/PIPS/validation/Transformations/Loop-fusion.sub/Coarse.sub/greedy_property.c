#include <stdlib.h>


int main(int argc, char**argv) {
  int n = argc>1 ? atoi(argv[1]) : 1000;
  int m = argc>2 ? atoi(argv[2]) : 5;
  if(n>1) {
    float h[1+m];
    float x[n+m];
    float y[n+m];
    int LU_IB0, i, m3, m4;
    LU_IB0 = n % 4;


    for(i = LU_IB0; i <= n-1; i += 4) {
      y[i+m] = 0;
      for(m4 = 0; m4 <= m-1; m4 += 1) {
        y[i+m] += 0;
      }
      y[i+m+1] = 0;
      for(m3 = 0; m3 <= m-1; m3 += 1) {
        y[i+m+1] += 0;
      }
    }
  }
  return 0;
}
