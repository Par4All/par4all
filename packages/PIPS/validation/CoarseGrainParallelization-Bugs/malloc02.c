/* Check the parallelization when thread-safe variables are ignored.
 *
 * Here, a is a pointer to pointer instead of an array of pointers as
 * in malloc01.c
 */

#define N 64

#include <malloc.h>

int main(int argc, char *argv[]) {
  int i, j;
  float ** a;

  a = malloc(N*4/*sizeof(float *)*/);
  for(i=0;i<N;i++) {
    a[i] = malloc(N*4/*sizeof(float)*/);
    for(j=0;j<N;j++) {
      a[i][j]=0.;
    }
  }

  return 0;
}
