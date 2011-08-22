/* Array region a is found empty in the outermost loop and in the
   module statement */

#include <stdio.h>

int main() {

  int N = 10;
  int a[N][N];


  for(int i=1; i<=N;i++) {
    for(int j=1; j<=N;j++) {
      if(i*j<N/2) {
        a[N-1-i][i+j-1] = 1;
        a[i-1][N-i-j-1] = 1;
	printf("i=%d, j=%d\n", i, j);
      }
      if(i==j) {
        a[i-1][j-1] = 1;
	printf("i=%d, j=%d\n", i, j);
      }
    }
  }
}
