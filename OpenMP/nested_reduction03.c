#include <stdio.h>

int size = 100;

int main (void) {
  int k = 0, i = 0, l = 0;
  int sum = 0, tmp = 0;

  for (l = 0; l < size; l++) {
    tmp = 0;
    for (k = 0; k < size; k++) {
      for (i = 0; i < size; i++) {
	tmp += i + k ;
      }
    }
    sum += tmp;
  }

  printf ("sum is %d", sum);

  return 0;
}
