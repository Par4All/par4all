// return the index of the first negative element, size if all positive
int find_neg (int size, int a[size]) {
  int i = 0;
  for (i = 0; i < size; i++) {
    if (a[i] < 0) break;
  }
  return i;
}
