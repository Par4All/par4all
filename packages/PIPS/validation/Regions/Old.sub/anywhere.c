// validates OUT regions intersection and difference operators
// even when there are anywhere effects

#include <stdio.h>
#define N 1
char *name = "";
int j = 0, k = 0; // Loop indices
int a[N];
int b[N];
int tmp;


void out1() {
  b[0] = 0;
  for (j = 0; j < N; j += 1) {
    a[j] = 1;
    for (k = 0; k < N; k += 1)
      // There is an out region here
      b[k] = a[j];
  }
  printf("%d", b[0]);  // use of b[0] must generate an OUT region before!
}

void out2() {
  b[0] = 0;
  for (j = 0; j < N; j += 1) {
    a[j] = 1;
    for (k = 0; k < N; k += 1)
      // There must also be an OUT region here!
      b[k] = a[j];
  }
  printf("%d", b[0]);  // use of b[0] must generate an out region before!

  /* This printf generates an ANYWHERE effect that
     used to make OUT regions buggy  */
  printf("%s", name);

}


void out3() {
  b[0] = 0;
  for (j = 0; j < N; j += 1) {
    for (k = 0; k < N; k += 1)
      // without read on a[j], there's no issue with the OUT region
      b[k] = 0;
  }
  printf("%d", b[0]);  // use of b[0]: must generate an OUT region before!

  /* This printf generates an ANYWHERE effect that
     used to make OUT regions buggy ... sometimes but not here!  */
  printf("%s", name);

}



