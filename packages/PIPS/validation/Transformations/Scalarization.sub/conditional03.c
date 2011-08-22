/* Many overflows and then a long wait...

   Same as conditional02, but with a proper use of array a

   Very time consuming... An interesting case of output region...
 */

#include <stdio.h>

int main(int argc, char ** argv) {

  int N = argc;
  int a[N][N];


  for(int i=1; i<=N;i++) {
    for(int j=1; j<=N;j++) {
      if(i*j<N/2) {
        a[N-1-i][i+j-1] = 1;
        a[i-1][N-i-j-1] = 1;
      }
      if(i==j) {
        a[i-1][j-1] = 1;
      }
    }
  }
  printf("%d\n", a[0][0]);
}
