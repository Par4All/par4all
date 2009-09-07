/* Check the parallelization when thread-safe variables are ignored. */

#define N 64

#include <malloc.h>

int main(int argc, char *argv[]) {
  int i, j;
  float * a[N];

  for(i=0;i<N;i++) {
    a[i] = malloc(N*4/*sizeof(float)*/);
    for(j=0;j<N;j++) {
      a[i][j]=0.;
    }
  }

  return 0;
}
