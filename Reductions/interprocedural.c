int foo1 (int i, int j) {
  return i +=j;
}

void foo2 (int* i, int j) {
  (*i) += j;
}

int main (void) {
  int k =0;
  int sum1 = 0;
  int sum2 = 0;

  for (k = 0; k < 100; k++) {
    sum1 = foo1 (sum1, k);
  }

  for (k = 0; k < 100; k++) {
    foo2 (&sum2, k);
  }

  return 0;
}
