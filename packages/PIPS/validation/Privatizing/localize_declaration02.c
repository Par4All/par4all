#include <stdio.h>

enum { N = 1000000 };

int main(int argc, char *argv[]) {
  float b[N], c[N], a;
  int i;

  a = 0;

 init:
  do {
    for(i = 0; i < N; ++i) {
      b[i]=i;
      c[i]=i+1;
    }
  } while(1);
 compute:
  do {
    for(i = 0; i < N; ++i)
      a = a + b[i] * c[i] ;
  } while(1);

  printf("Result = %f\n", a);
  return 0;
}
