// return the index of the first negative element, -1 if all positive
int find_neg (int size, int a[size]) {
  int i = 0;
  int result = -1;

  // this loop must not be parallelized
  for (i = 0; i < size; i++) {
    if (a[i] < 0) return i;
  }
  return result;
}

int main () {
  int size = 10, i =0;
  int a[size];
  for (i = 0; i < size; i++) {
    a[i] = 0;
  }
  return find_neg (size, a);
}
