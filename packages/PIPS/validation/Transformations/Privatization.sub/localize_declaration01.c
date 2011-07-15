#include <stdio.h>

enum { N = 1000000 };

int main(int argc, char *argv[]) {
  float b[N], c[N], a;
  int i;

  a = 0;

 init:
  for(i = 0; i < N; ++i) {
    b[i]=i;
    c[i]=i+1;
  }
 compute:
  for(i = 0; i < N; ++i)
        a = a + b[i] * c[i] ;

  printf("Result = %f\n", a);
  return 0;
}
