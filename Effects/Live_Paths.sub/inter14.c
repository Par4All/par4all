// update a sub part of an array

int SIZE = 10;

void init (int a[SIZE], int val) {
  int i = 0;
  for (i =0; i < SIZE; i++)
    a[i] = val;
}

void foo (int a[SIZE], int min, int max, int val) {
  int i = 0;
  for (i = min; (i < max) && (i < SIZE); i++)
    a[i] += val;
}

int main () {
  int a[10];

  init (a, 0);
  foo (a, 3, 20, 20);

  return a[5];
}
