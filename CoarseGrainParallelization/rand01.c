/* Check the parallelization when the random number generator is called. */

#define N 64

int main(int argc, char *argv[]) {
  int i, j;
  float a[N][N];

  for(i=0;i<N;i++) {
    for(j=0;j<N;j++) {
      a[i][j]= rand();
    }
  }

  return 0;
}
