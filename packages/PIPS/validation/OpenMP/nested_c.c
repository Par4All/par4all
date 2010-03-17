#include <stdio.h>

int main (void) {
  int k =0, i = 0;
  int sum = 0;

  for (k = 0; k < 100; k++) {
    for (i = 0; i < 100; i++) {
      sum += i;
    }
    sum += k;
  }

  for (i = 0; i < 100; i++) {
    sum += i;
  }

  printf ("sum: %d\n", sum);

  return 0;
}
