// initialize a sub part of an array

void foo (int size, int a[size], int min, int max, int val) {
  int i = 0;
  for (i = min; (i < max) && (i < size); i++)
    a[i] = val;
}

int main () {
  int a[10];
  int i;

  foo (10, a, 3, 20, 20);

  return a[5];
}
