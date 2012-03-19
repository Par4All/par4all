// initialize a sub part of an array passed by pointer

void foo (int size, int* ptr, int min, int max, int val) {
  int i = 0;
  for (i = min; (i < max) && (i < size); i++)
    ptr[i] = val;
}

int main () {
  int a[10];

  foo (10, a, 3, 20, 20);

  return a[5];
}
