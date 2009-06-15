#include <stdio.h>

int foo1 (int i, int j) {
  return i +=j;
}

void foo2 (int* i, int j) {
  (*i) += j;
}

int foo3 (int i, int j) {
  static int k = 0;
  i +=j;
  return (k = i);
}

int foo4 (int i, int j) {
  int k = 0;
  i +=j;
  return (k = i);
}

int nontrivial (void) {
  int i = 0, b = 0;
  for (i=0; i < 100; i++) {
    b = b + b + i;
  }
  return b;
}

int main (void) {
  int k =0;
  int sum1 = 0;
  int sum2 = 0;
  int sum3 = 0;
  int sum4 = 0;
  int sum5 = 0;

  for (k = 0; k < 100; k++) {
    sum1 = foo1 (sum1, k);
  }

  for (k = 0; k < 100; k++) {
    foo2 (&sum2, k);
  }

  for (k = 0; k < 100; k++) {
    sum3 = foo3 (sum3, k);
  }

  for (k = 0; k < 100; k++) {
    sum4 = foo4 (sum4, k);
  }

  for (k = 0; k < 100; k++) {
    sum5 += k;
  }

  printf ("sum1: %d\nsum2: %d\nsum3: %d\nsum4: %d\nsum5: %d\n", sum1, sum2,
	  sum3, sum4, sum5);

  return 0;
}
