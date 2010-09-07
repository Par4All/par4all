// return the index of the first negative element, -1 if all positive
int find_neg (int size, int a[size]) {
  int i = 0;
  int result = -1;
  for (i = 0; i < size; i++) {
    if (a[i] < 0) return i;
  }
  return result;
}
