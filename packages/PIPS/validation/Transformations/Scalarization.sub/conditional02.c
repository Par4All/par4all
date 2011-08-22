/* Many overflows and then a long wait...

   Since a is never used, everything is scalarizable... A printf() of
   array a should be added for more reasonnable results. See
   conditional03.c
 */


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
}
